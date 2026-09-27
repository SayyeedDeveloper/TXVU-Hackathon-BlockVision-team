"""src/pick_points.py — interactive point-picker for scene geometry.

Manually eyeballing pixel coordinates off a still frame (as the first pass in
src/config.py's SCENE did) is too imprecise — this lets you click directly on
the frame instead and prints ready-to-paste LaneDirection/CrossingZone/
StopLine code with normalized coordinates.

One-off inspection tool, not part of the submission pipeline. Only needs cv2
(+ numpy); no ultralytics/torch.

Usage:
    python -m src.pick_points
    python -m src.pick_points --frame samples/frames/C3896/frame_00.jpg

Flow (type is chosen BEFORE you start clicking, so the tool knows exactly how
many clicks to expect and can auto-finish the shape — no more "click points
then press a key to finish" ambiguity):
    1. Terminal prompts: "shape type (lane/crossing/stop_line), 'l' to list,
       'x' to delete, or 'q' to finish: " then "shape name: ".
    2. Click on the image window:
         - stop_line: exactly 2 clicks (start, end) -> auto-finishes.
         - crossing:  exactly 4 clicks (corners)     -> auto-finishes.
         - lane:      as many clicks as you need to trace the boundary
                      (straight lanes: 4 corners; curving/turning lanes: as
                      many as it takes) -> press 'd' or Enter when done (at
                      least 3 points), then asks for a direction: type
                      "dx,dy" (e.g. "1,0") in the terminal, or press enter
                      and click a start + end point on the image.
    3. Repeats from 1. 'q' at the type prompt (or in the image window during
       clicking) ends the session, discards any not-yet-clicked shape,
       prints the ready-to-paste Python, and writes points_check.jpg.

Controls while clicking a shape:
    left click      add the next point
    u               undo the last point
    r               reset the current shape's points (start it over)
    d / Enter        lane only: finish the polygon (needs >= 3 points);
                     crossing/stop_line auto-finish on click count instead
    q               abort and quit the whole session

Why this used to freeze on 'n': the old flow called Python's blocking
input() directly from inside the same single thread that also owns the cv2
window (the thread that must keep calling cv2.waitKey() for the window to
stay responsive to the OS). The instant input() was reached, that thread
stopped calling cv2.waitKey() entirely until you finished typing and hit
Enter — on macOS in particular, a Cocoa-backed window that goes any real
length of time without its owning thread pumping events gets flagged
"Application Not Responding" and starts fighting you for keyboard/mouse
focus, which is exactly what looked like a freeze. Every terminal prompt now
goes through `threaded_input()`, which runs input() on a background thread
while the main thread keeps calling cv2.waitKey()+redraw() in a tight loop
for as long as the prompt is unanswered — the window is never starved of
event pumping, no matter how long you take to type. See threaded_input()'s
own docstring and tests/test_pick_points.py for the automated check that
this is actually happening (the pump function is proven to keep firing
while input() blocks), not just a hopeful reduction in likelihood.

The image is shown scaled to fit the screen if it's larger than
MAX_DISPLAY_W x MAX_DISPLAY_H; every click is converted back to full-
resolution coordinates immediately, and the scale factor used is printed at
startup. All arithmetic inside PointPicker operates in full-resolution pixel
space — only rendering for the on-screen window scales down.
"""
from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.rules.geometry import unit_vector

MAX_DISPLAY_W = 1600
MAX_DISPLAY_H = 900

GREEN = (0, 200, 0)      # lane (newly picked this session)
BLUE = (255, 120, 0)     # crossing (newly picked this session)
RED = (0, 0, 255)        # stop_line (newly picked this session)
ORANGE = (0, 140, 255)   # traffic_light_roi (newly picked this session)
YELLOW = (0, 220, 220)   # in-progress shape (main click buffer)
MAGENTA = (200, 0, 200)  # in-progress lane-direction pick

# Overlay colors for geometry ALREADY in src/config.py (drawn as a read-only
# backdrop at startup so you can see what's picked before adding more). Uses
# the live visualizer's palette (see SCENE_*_COLOR in src/visualize_trails.py),
# deliberately different hues from the picker's own new-shape colors above so
# "already saved" vs "just picked this session" stay visually distinct.
EXIST_LANE_COLOR = (255, 180, 0)      # blue
EXIST_CROSSING_COLOR = (255, 0, 255)  # magenta
EXIST_STOPLINE_COLOR = (0, 255, 255)  # yellow
EXIST_TL_COLOR = (0, 140, 255)        # orange

VALID_KINDS = ("lane", "crossing", "stop_line", "traffic_light_roi")
# "lane" is intentionally absent: its polygon is variable-length (see
# collect_polygon_clicks), unlike crossing/stop_line/traffic_light_roi which
# stay fixed-count.
REQUIRED_CLICKS = {"crossing": 4, "stop_line": 2, "traffic_light_roi": 2}
LANE_MIN_POINTS = 3
LANE_DONE_KEYS = ("d", "\r", "\n")


def normalize_point(pt: tuple[float, float], img_w: int, img_h: int) -> tuple[float, float]:
    """Pixel coords -> normalized [0,1] fractions, rounded to 3 decimals --
    the exact convention CrossingZone/LaneDirection/StopLine polygons use in
    src/config.py. Factored out here (not just duplicated) so other geometry
    input paths -- e.g. automatic detections from another tool --
    produce coordinates in the identical format/rounding as hand-picked ones,
    rather than reinventing normalization with potentially different rounding."""
    return (round(pt[0] / img_w, 3), round(pt[1] / img_h, 3))


@dataclass
class Shape:
    kind: str                              # "lane" | "crossing" | "stop_line" | "traffic_light_roi"
    name: str
    points: list[tuple[int, int]]          # full-resolution pixel coords
    direction: tuple[float, float] | None = None  # unit vector, "lane" only
    lane_name: str | None = None           # "stop_line" only -- StopLine.lane_name


class PointPicker:
    """Pure state machine, independent of any GUI event loop — this is what
    gets exercised directly in a smoke test without opening a real window."""

    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h
        self.current: list[tuple[int, int]] = []
        self.shapes: list[Shape] = []

    def add_point(self, x: int, y: int) -> None:
        self.current.append((int(x), int(y)))

    def undo(self) -> None:
        if self.current:
            self.current.pop()

    def reset_current(self) -> None:
        self.current = []

    def finish_shape(
        self,
        name: str,
        kind: str,
        direction: tuple[float, float] | None = None,
        lane_name: str | None = None,
    ) -> Shape:
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        if not self.current:
            raise ValueError("no points in the current shape")
        if kind in ("stop_line", "traffic_light_roi") and len(self.current) > 2:
            print(f"  note: {kind} got {len(self.current)} points, using only the first 2")
        pts = self.current[:2] if kind in ("stop_line", "traffic_light_roi") else list(self.current)
        shape = Shape(kind=kind, name=name, points=pts, direction=direction, lane_name=lane_name)
        self.shapes.append(shape)
        self.current = []
        return shape

    def list_shapes(self) -> list[tuple[int, str, str]]:
        """(index, name, kind) for every completed shape, in the order 'x' expects."""
        return [(i, s.name, s.kind) for i, s in enumerate(self.shapes)]

    def delete_shape(self, index: int) -> Shape:
        if not (0 <= index < len(self.shapes)):
            raise IndexError(f"no shape at index {index} (have {len(self.shapes)})")
        return self.shapes.pop(index)

    # ---- output -----------------------------------------------------
    # maps a picked shape's `kind` to the SceneConfig(...) list kwarg it belongs in.
    _CONFIG_LIST_KEY = {
        "lane": "lanes",
        "crossing": "crossings",
        "stop_line": "stop_lines",
        "traffic_light_roi": "traffic_lights",
    }

    def _norm(self, pt: tuple[int, int]) -> tuple[float, float]:
        return normalize_point(pt, self.img_w, self.img_h)

    def _entry_source(self, s: Shape) -> str:
        """One constructor-call string for shape `s`, with NO trailing comma
        and NO inline comment. The single source of truth for what a picked
        shape looks like as code -- to_python_code() builds its copy-pasteable
        terminal block from these."""
        if s.kind == "lane":
            poly = ", ".join(str(self._norm(p)) for p in s.points)
            d = s.direction or (1.0, 0.0)
            return (f'LaneDirection(name="{s.name}", polygon=[{poly}], '
                    f'direction=({round(d[0], 3)}, {round(d[1], 3)}))')
        if s.kind == "crossing":
            poly = ", ".join(str(self._norm(p)) for p in s.points)
            return f'CrossingZone(name="{s.name}", polygon=[{poly}])'
        if s.kind == "stop_line":
            if len(s.points) >= 2:
                p1, p2 = self._norm(s.points[0]), self._norm(s.points[1])
            else:
                p1 = p2 = self._norm(s.points[0])
            lane_name = f'"{s.lane_name}"' if s.lane_name else "None"
            return f'StopLine(name="{s.name}", p1={p1}, p2={p2}, approach_side=1, lane_name={lane_name})'
        if s.kind == "traffic_light_roi":
            if len(s.points) >= 2:
                (x1, y1), (x2, y2) = self._norm(s.points[0]), self._norm(s.points[1])
            else:
                (x1, y1), (x2, y2) = self._norm(s.points[0]), self._norm(s.points[0])
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return f'TrafficLightROI(name="{s.name}", box={box})'
        raise ValueError(f"unknown kind {s.kind!r}")

    def to_entries_by_kind(self) -> dict[str, list[str]]:
        """{config_list_key: [entry_source, ...]} for every shape picked this
        session, grouped by the SceneConfig list it belongs in. Consumed by
        merge_into_config() to APPEND these. Kinds with no picked shapes are
        simply absent from the dict."""
        out: dict[str, list[str]] = {}
        for s in self.shapes:
            out.setdefault(self._CONFIG_LIST_KEY[s.kind], []).append(self._entry_source(s))
        return out

    def to_python_code(self, source: str = "") -> str:
        by_key: dict[str, list[Shape]] = {
            "lanes": [s for s in self.shapes if s.kind == "lane"],
            "crossings": [s for s in self.shapes if s.kind == "crossing"],
            "stop_lines": [s for s in self.shapes if s.kind == "stop_line"],
            "traffic_lights": [s for s in self.shapes if s.kind == "traffic_light_roi"],
        }
        lines: list[str] = []
        if source:
            lines.append(f"# picked from {source} ({self.img_w}x{self.img_h})")
        for key in ("lanes", "crossings", "stop_lines", "traffic_lights"):
            lines.append(f"{key}=[")
            for s in by_key[key]:
                # the sign-convention TODO is display-only (kept out of _entry_source
                # so it never lands inside a spliced list literal as a stray comment)
                suffix = ("  # TODO: verify sign against verify_scene.py overlay, flip to -1 if backwards"
                          if s.kind == "stop_line" else "")
                lines.append(f"    {self._entry_source(s)},{suffix}")
            lines.append("],")
        return "\n".join(lines)

    # ---- rendering ----------------------------------------------------
    def render(
        self,
        canvas: np.ndarray,
        scale: float = 1.0,
        extra_pending: list[tuple[float, float]] | None = None,
        extra_color: tuple[int, int, int] = MAGENTA,
    ) -> np.ndarray:
        """Draw all finished shapes + the in-progress one (self.current, in
        YELLOW) onto `canvas` (already at whatever resolution `canvas` is;
        `scale` converts this picker's full-resolution stored points down to
        canvas space). `extra_pending`, if given, is a SEPARATE live buffer
        drawn in `extra_color` — used for the lane-direction click phase,
        which runs after the polygon is done but before finish_shape() is
        called, so it must render independently of self.current."""
        out = canvas

        def to_disp(pt):
            return (int(pt[0] * scale), int(pt[1] * scale))

        color_by_kind = {"lane": GREEN, "crossing": BLUE, "stop_line": RED, "traffic_light_roi": ORANGE}
        for shape in self.shapes:
            color = color_by_kind[shape.kind]
            pts = [to_disp(p) for p in shape.points]
            if shape.kind == "traffic_light_roi":
                if len(pts) >= 2:
                    (x1, y1), (x2, y2) = pts[0], pts[1]
                    cv2.rectangle(out, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), color, 2)
            else:
                closed = shape.kind != "stop_line"
                if len(pts) >= 2:
                    cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=closed, color=color, thickness=2)
            for p in pts:
                cv2.circle(out, p, 4, color, -1)
            if pts:
                cv2.putText(out, shape.name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if shape.kind == "lane" and shape.direction and pts:
                cx = int(sum(p[0] for p in pts) / len(pts))
                cy = int(sum(p[1] for p in pts) / len(pts))
                dx, dy = shape.direction
                tip = (int(cx + dx * 60), int(cy + dy * 60))
                cv2.arrowedLine(out, (cx, cy), tip, color, 3, tipLength=0.3)

        def draw_pending(pts_full: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
            pts = [to_disp(p) for p in pts_full]
            for i, p in enumerate(pts):
                cv2.circle(out, p, 5, color, -1)
                cv2.putText(out, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if len(pts) >= 2:
                cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=1)

        draw_pending(self.current, YELLOW)
        if extra_pending:
            draw_pending(extra_pending, extra_color)
        return out


def draw_existing_geometry(canvas: np.ndarray, scene, disp_w: int, disp_h: int) -> int:
    """Draw geometry ALREADY saved in `scene` (an imported SceneConfig) onto
    `canvas` (a disp_w x disp_h display image) as a read-only backdrop. Coords
    in `scene` are full-resolution pixels (SceneConfig.__post_init__ scaled
    them by scene.width/height); this normalizes them back out and rescales to
    the display size, so it's correct regardless of the loaded frame's
    resolution -- exactly how src/visualize_trails.py's draw_scene_geometry()
    does it. Returns how many shapes were drawn. Does NOT touch any
    PointPicker state, so nothing drawn here can ever be re-saved."""
    sw = getattr(scene, "width", 0) or disp_w
    sh = getattr(scene, "height", 0) or disp_h

    def to_disp(pt: tuple[float, float]) -> tuple[int, int]:
        return (int(pt[0] / sw * disp_w), int(pt[1] / sh * disp_h))

    n = 0
    for lane in getattr(scene, "lanes", []):
        pts = np.array([to_disp(p) for p in lane.polygon], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=True, color=EXIST_LANE_COLOR, thickness=2)
        if len(pts):
            cv2.putText(canvas, lane.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EXIST_LANE_COLOR, 1)
        n += 1
    for c in getattr(scene, "crossings", []):
        pts = np.array([to_disp(p) for p in c.polygon], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=True, color=EXIST_CROSSING_COLOR, thickness=2)
        if len(pts):
            cv2.putText(canvas, c.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EXIST_CROSSING_COLOR, 1)
        n += 1
    for sl in getattr(scene, "stop_lines", []):
        cv2.line(canvas, to_disp(sl.p1), to_disp(sl.p2), EXIST_STOPLINE_COLOR, 3)
        cv2.putText(canvas, sl.name, to_disp(sl.p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EXIST_STOPLINE_COLOR, 1)
        n += 1
    for tl in getattr(scene, "traffic_lights", []):
        x1, y1, x2, y2 = tl.box
        (px1, py1), (px2, py2) = to_disp((x1, y1)), to_disp((x2, y2))
        cv2.rectangle(canvas, (px1, py1), (px2, py2), EXIST_TL_COLOR, 2)
        cv2.putText(canvas, tl.name, (px1, max(0, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EXIST_TL_COLOR, 1)
        n += 1
    return n


# SceneConfig list-attr -> (config_list_key, display kind) for existing shapes.
# Order here defines the startup numbering (matches draw_existing_geometry above).
_EXISTING_ATTRS = [
    ("lanes", "lane"),
    ("crossings", "crossing"),
    ("stop_lines", "stop_line"),
    ("traffic_lights", "traffic_light"),
]


def load_existing_shapes(scene) -> list[dict]:
    """Flatten a SceneConfig's already-saved geometry into an ordered list of
    {'config_key','name','kind'} -- one per lane/crossing/stop_line/traffic_light,
    in the same order draw_existing_geometry() draws them. This is what the
    startup numbered list and the 'x'-delete feature index into; each entry's
    (config_key, name) is exactly what merge_into_config() needs to drop it."""
    out: list[dict] = []
    for attr, kind in _EXISTING_ATTRS:
        for obj in getattr(scene, attr, []) or []:
            out.append({"config_key": attr, "name": obj.name, "kind": kind})
    return out


def merge_into_config(
    entries_by_kind: dict[str, list[str]],
    deletions_by_kind: dict[str, set] | None = None,
    config_path: str = "src/config.py",
) -> list[str]:
    """Rewrite the `SCENE = SceneConfig(...)` call in `config_path` IN PLACE:
    for each affected list kwarg, drop existing elements whose `name=` is in
    `deletions_by_kind[key]`, keep the rest verbatim (source text preserved via
    ast.get_source_segment), then append the new `entries_by_kind[key]` strings.
    Lists/comments not touched by a deletion or addition are left byte-for-byte
    identical. A list kwarg that doesn't exist yet is created (for additions).
    Re-parses the result before writing, so a bug can never leave a broken file.
    Returns human-readable descriptions of what changed."""
    import ast

    entries_by_kind = {k: v for k, v in (entries_by_kind or {}).items() if v}
    deletions_by_kind = {k: set(v) for k, v in (deletions_by_kind or {}).items() if v}
    if not entries_by_kind and not deletions_by_kind:
        return []

    path = Path(config_path)
    src = path.read_text()
    tree = ast.parse(src)

    scene_call = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SCENE" for t in node.targets)
                and isinstance(node.value, ast.Call)):
            scene_call = node.value
            break
    if scene_call is None:
        raise RuntimeError(f"could not find `SCENE = SceneConfig(...)` in {config_path}")

    line_start = [0]
    for ln in src.splitlines(keepends=True):
        line_start.append(line_start[-1] + len(ln))

    def off(lineno: int, col: int) -> int:
        return line_start[lineno - 1] + col

    def elt_name(el) -> str | None:
        """The name="..." of a constructor-call list element, if any."""
        if isinstance(el, ast.Call):
            for kw in el.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
        return None

    kwargs = {kw.arg: kw for kw in scene_call.keywords if kw.arg}
    INDENT = "        "

    edits: list[tuple[int, int, str]] = []
    close_inserts: list[str] = []
    described: list[str] = []

    for key in set(entries_by_kind) | set(deletions_by_kind):
        dels = deletions_by_kind.get(key, set())
        adds = entries_by_kind.get(key, [])
        if key in kwargs and isinstance(kwargs[key].value, ast.List):
            lst = kwargs[key].value
            kept, dropped = [], []
            for el in lst.elts:
                nm = elt_name(el)
                if nm is not None and nm in dels:
                    dropped.append(nm)
                else:
                    kept.append(ast.get_source_segment(src, el))
            all_entries = kept + adds
            if all_entries:
                body = "".join(f"{INDENT}{e},\n" for e in all_entries)
                replacement = "[\n" + body + "    ]"
            else:
                replacement = "[]"
            edits.append((off(lst.lineno, lst.col_offset),
                          off(lst.end_lineno, lst.end_col_offset), replacement))
            if dropped:
                described.append(f"{key}: deleted {dropped}")
            if adds:
                described.append(f"{key}: appended {len(adds)} new")
        else:
            # kwarg missing: only additions are meaningful (nothing to delete)
            if adds:
                body = "".join(f"{INDENT}{e},\n" for e in adds)
                close_inserts.append(f"    {key}=[\n" + body + "    ],\n")
                described.append(f"{key}: created new list with {len(adds)}")

    if close_inserts:
        close_off = off(scene_call.end_lineno, scene_call.end_col_offset) - 1
        edits.append((close_off, close_off, "".join(close_inserts)))

    for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        src = src[:start] + rep + src[end:]

    ast.parse(src)  # never write a file that doesn't parse
    path.write_text(src)
    return described



def _parse_direction(text: str) -> tuple[float, float] | None:
    text = text.strip()
    if "," not in text:
        return None
    try:
        dx_str, dy_str = text.split(",", 1)
        dx, dy = float(dx_str), float(dy_str)
    except ValueError:
        return None
    return unit_vector((dx, dy))


def threaded_input(prompt: str, pump_fn: Callable[[], None], poll_sec: float = 0.03) -> str:
    """input() on a background daemon thread while `pump_fn` keeps getting
    called on THIS (the caller's/GUI's) thread every `poll_sec` seconds until
    the answer arrives. This is the actual freeze fix: the previous version
    called input() directly on the thread that owns the cv2 window, which
    meant that thread stopped calling cv2.waitKey() for as long as the
    terminal prompt sat unanswered. Blocking stdin reads release the GIL, so
    the background thread genuinely doesn't block this one — pump_fn keeps
    firing on a ~30ms cadence for however long the user takes to type.
    """
    result_q: "queue.Queue[str]" = queue.Queue()

    def worker() -> None:
        try:
            result_q.put(input(prompt))
        except EOFError:
            result_q.put("q")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        pump_fn()
        t.join(timeout=poll_sec)
    return result_q.get()


def collect_clicks(
    n: int,
    add_fn: Callable[[tuple[float, float]], None],
    undo_fn: Callable[[], None],
    reset_fn: Callable[[], None],
    len_fn: Callable[[], int],
    key_fn: Callable[[], str],
    pump_fn: Callable[[], None],
) -> bool:
    """Wait for exactly `n` points to accumulate (added asynchronously by
    something else — the real mouse callback, or a test driver — via
    `add_fn`), while polling `key_fn` each tick for u/r/q. Never calls
    input(): key-only, so it never overlaps with threaded_input()'s prompts.
    Returns False (and resets) if the user aborted with 'q'.
    """
    aborted = False
    while len_fn() < n:
        ch = key_fn()
        if ch == "q":
            aborted = True
            break
        elif ch == "u":
            undo_fn()
        elif ch == "r":
            reset_fn()
        pump_fn()
    if aborted:
        reset_fn()
    return not aborted


def collect_polygon_clicks(
    add_fn: Callable[[tuple[float, float]], None],
    undo_fn: Callable[[], None],
    reset_fn: Callable[[], None],
    len_fn: Callable[[], int],
    key_fn: Callable[[], str],
    pump_fn: Callable[[], None],
    min_points: int = LANE_MIN_POINTS,
    done_keys: tuple[str, ...] = LANE_DONE_KEYS,
) -> bool:
    """Like collect_clicks, but variable-length: keep collecting points
    (added asynchronously via `add_fn`, same as collect_clicks) until the
    caller presses one of `done_keys` with at least `min_points` collected,
    or aborts with 'q'. Used for lane polygons so a curving/turning lane can
    have as many corners as it actually needs instead of a fixed 4. Same
    key-only contract as collect_clicks: never calls input(), so it never
    overlaps with threaded_input()'s prompts.
    """
    aborted = False
    while True:
        ch = key_fn()
        if ch == "q":
            aborted = True
            break
        elif ch == "u":
            undo_fn()
        elif ch == "r":
            reset_fn()
        elif ch in done_keys:
            if len_fn() >= min_points:
                break
            print(f"  need at least {min_points} points before finishing (have {len_fn()})")
        pump_fn()
    if aborted:
        reset_fn()
    return not aborted


def run_interactive(frame_path: str) -> int:
    img = cv2.imread(frame_path)
    if img is None:
        print(f"could not read {frame_path}")
        return 1
    img_h, img_w = img.shape[:2]
    scale = min(1.0, MAX_DISPLAY_W / img_w, MAX_DISPLAY_H / img_h)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    base_disp = cv2.resize(img, (disp_w, disp_h)) if scale < 1.0 else img.copy()
    print(f"loaded {frame_path} ({img_w}x{img_h}); display scale = {scale:.4f} ({disp_w}x{disp_h})")

    # Bake already-saved geometry from src/config.py into base_disp as a
    # read-only backdrop -- shown every redraw (base_disp is copied, never
    # mutated by the picker) but never part of picker.shapes, so on 'q' only
    # the shapes picked THIS session get appended. Import is deferred to here
    # so the tool still opens on a broken/empty config (just with no backdrop).
    existing_shapes: list[dict] = []
    try:
        from src.config import SCENE as _existing_scene
        n_existing = draw_existing_geometry(base_disp, _existing_scene, disp_w, disp_h)
        existing_shapes = load_existing_shapes(_existing_scene)
        if n_existing:
            print(f"overlaid {n_existing} already-picked shape(s) from src/config.py "
                  f"(backdrop). Newly-picked shapes are appended on quit; existing ones can "
                  f"be marked for deletion with 'x'.")
    except Exception as exc:
        print(f"warning: could not overlay existing src/config.py geometry ({exc!r}); "
              f"starting with just the frame")

    existing_deleted: set[int] = set()   # indices into existing_shapes marked for deletion this session

    def print_shape_menu() -> None:
        """Numbered list: existing config shapes (1..E, with a [MARKED FOR DELETION]
        flag) then shapes picked this session (E+1..). The 'x' command indexes into
        exactly this numbering."""
        print("\nshapes:")
        if existing_shapes:
            print("  existing (in src/config.py):")
            for i, s in enumerate(existing_shapes):
                flag = "  [MARKED FOR DELETION]" if i in existing_deleted else ""
                print(f"    {i + 1}. {s['name']} ({s['kind']}){flag}")
        else:
            print("  existing (in src/config.py): none")
        if picker.shapes:
            print("  newly picked this session:")
            for j, s in enumerate(picker.shapes):
                print(f"    {len(existing_shapes) + j + 1}. {s.name} ({s.kind})  [new]")
    if existing_shapes:
        print(f"\n{len(existing_shapes)} existing shape(s) loaded from src/config.py:")
        for i, s in enumerate(existing_shapes):
            print(f"  {i + 1}. {s['name']} ({s['kind']})")

    picker = PointPicker(img_w, img_h)
    win = "pick_points  (click=add point  u=undo  r=reset  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    click_sink = {"accepting": False, "add_fn": None, "extra_pending": None}

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or not click_sink["accepting"]:
            return
        orig = (x / scale, y / scale)
        click_sink["add_fn"](orig)

    cv2.setMouseCallback(win, on_mouse)

    def redraw() -> None:
        canvas = base_disp.copy()
        picker.render(canvas, scale=scale, extra_pending=click_sink["extra_pending"])
        cv2.imshow(win, canvas)

    def cv_key_fn() -> str:
        key = cv2.waitKey(30) & 0xFF
        return chr(key) if key < 128 and key != 255 else ""

    def cv_pump_fn() -> None:
        redraw()
        cv2.waitKey(1)

    def collect(collector_fn, add_fn, undo_fn, reset_fn, len_fn, extra_pending=None) -> bool:
        """`collector_fn` is collect_clicks or collect_polygon_clicks, already
        bound to everything except (add_fn, undo_fn, reset_fn, len_fn, key_fn, pump_fn)."""
        click_sink["accepting"] = True
        click_sink["add_fn"] = add_fn
        click_sink["extra_pending"] = extra_pending
        try:
            return collector_fn(add_fn, undo_fn, reset_fn, len_fn, cv_key_fn, cv_pump_fn)
        finally:
            click_sink["accepting"] = False
            click_sink["extra_pending"] = None

    redraw()
    quit_requested = False
    while not quit_requested:
        kind = ""
        while True:
            kind = threaded_input(
                f"\nshape type {VALID_KINDS}, 'l' to list shapes, 'x' to delete one, or 'q' to finish: ",
                cv_pump_fn,
            ).strip().lower()
            if kind == "l":
                print_shape_menu()
                continue
            if kind == "x":
                print_shape_menu()
                n_exist = len(existing_shapes)
                raw = threaded_input(
                    "delete which number? (existing -> toggle mark; new -> remove now; blank to cancel): ",
                    cv_pump_fn,
                ).strip()
                if raw:
                    try:
                        num = int(raw)
                        if 1 <= num <= n_exist:
                            idx = num - 1
                            s = existing_shapes[idx]
                            if idx in existing_deleted:
                                existing_deleted.discard(idx)
                                print(f"  unmarked existing {s['name']} ({s['kind']}) -- will be KEPT")
                            else:
                                existing_deleted.add(idx)
                                print(f"  marked existing {s['name']} ({s['kind']}) FOR DELETION on quit")
                        elif n_exist < num <= n_exist + len(picker.shapes):
                            removed = picker.delete_shape(num - n_exist - 1)
                            print(f"  removed newly-picked {removed.name} ({removed.kind})")
                        else:
                            print(f"  no shape numbered {num}")
                    except (ValueError, IndexError) as e:
                        print(f"  not deleted: {e}")
                    redraw()
                continue
            if kind in VALID_KINDS or kind == "q":
                break
            print(f"unrecognized: {kind!r}")

        if kind == "q":
            break

        name = threaded_input("shape name (e.g. 'near_side_eastbound'): ", cv_pump_fn).strip()

        if kind == "lane":
            print(f"click {LANE_MIN_POINTS}+ boundary point(s) for '{name}' on the image, "
                  f"then press 'd' or Enter when done (u=undo  r=reset  q=quit)...")
            collector = collect_polygon_clicks
        else:
            n_clicks = REQUIRED_CLICKS[kind]
            print(f"click {n_clicks} point(s) for '{name}' on the image (u=undo  r=reset  q=quit)...")
            collector = partial(collect_clicks, n_clicks)

        ok = collect(collector, lambda pt: picker.add_point(*pt), picker.undo, picker.reset_current,
                      lambda: len(picker.current))
        if not ok:
            print("quit requested, discarding this shape")
            quit_requested = True
            break

        direction = None
        if kind == "lane":
            raw = threaded_input(
                "direction: type 'dx,dy' (e.g. '1,0'), or press enter to click start+end points: ",
                cv_pump_fn,
            )
            direction = _parse_direction(raw)
            if direction is None:
                print("click a START point then an END point for the travel direction...")
                dir_pts: list[tuple[float, float]] = []
                ok = collect(partial(collect_clicks, 2), dir_pts.append,
                              lambda: dir_pts.pop() if dir_pts else None, dir_pts.clear,
                              lambda: len(dir_pts), extra_pending=dir_pts)
                if not ok:
                    print("quit requested, discarding this shape")
                    picker.reset_current()
                    quit_requested = True
                    break
                (sx, sy), (ex, ey) = dir_pts
                direction = unit_vector((ex - sx, ey - sy))

        lane_name = None
        if kind == "stop_line":
            raw = threaded_input(
                "which lane does this stop line belong to? (lane name, or blank for none): ", cv_pump_fn,
            ).strip()
            lane_name = raw or None

        picker.finish_shape(name, kind, direction=direction, lane_name=lane_name)
        print(f"  saved {kind} '{name}' with {len(picker.shapes[-1].points)} point(s)")
        redraw()

    cv2.destroyAllWindows()

    if picker.current:
        print(f"warning: {len(picker.current)} unfinished point(s) discarded")
        picker.reset_current()

    code = picker.to_python_code(source=frame_path)
    print("\n" + "=" * 70)
    print("shapes picked this session (also the additions in the merge below):")
    print("=" * 70)
    print(code)
    print("=" * 70)

    # ---- build the merge: append newly-picked shapes, drop marked-for-deletion
    #      existing ones. Everything else in config.py stays untouched. --------
    entries = picker.to_entries_by_kind()                    # {config_key: [new source str, ...]}
    deletions: dict[str, set] = {}
    for idx in existing_deleted:
        s = existing_shapes[idx]
        deletions.setdefault(s["config_key"], set()).add(s["name"])

    kept = [s for i, s in enumerate(existing_shapes) if i not in existing_deleted]
    print("\n" + "=" * 70)
    print("PENDING CHANGES to src/config.py:")
    print("=" * 70)
    print(f"  KEEP   ({len(kept)} existing): " +
          (", ".join(f"{s['name']}({s['kind']})" for s in kept) or "none"))
    print(f"  DELETE ({len(existing_deleted)} existing): " +
          (", ".join(f"{existing_shapes[i]['name']}({existing_shapes[i]['kind']})"
                     for i in sorted(existing_deleted)) or "none"))
    print(f"  ADD    ({len(picker.shapes)} new): " +
          (", ".join(f"{s.name}({s.kind})" for s in picker.shapes) or "none"))
    print("=" * 70)

    if not entries and not deletions:
        print("no additions or deletions -- src/config.py left untouched")
    else:
        # window is already destroyed, so a plain blocking input() is safe here
        # (no cv2 event loop to starve -- that was only a concern while picking)
        resp = input("write these changes to src/config.py now? [y/N]: ").strip().lower()
        if resp == "y":
            try:
                changed = merge_into_config(entries, deletions, "src/config.py")
                print("updated src/config.py:")
                for line in changed:
                    print(f"  - {line}")
            except Exception as exc:
                print(f"ERROR writing src/config.py ({exc!r}); nothing was changed. "
                      f"Paste the block above manually if needed.")
        else:
            print("not written -- paste the block above manually to apply additions "
                  "(deletions would need to be removed by hand).")

    out_path = Path(frame_path).with_name("points_check.jpg")
    check_img = img.copy()
    picker.render(check_img, scale=1.0)
    cv2.imwrite(str(out_path), check_img)
    print(f"\nwrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="samples/frames/C3896/frame_00.jpg")
    args = ap.parse_args()
    return run_interactive(args.frame)


if __name__ == "__main__":
    raise SystemExit(main())
