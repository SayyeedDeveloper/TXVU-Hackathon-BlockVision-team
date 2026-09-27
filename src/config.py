"""src/config.py — scene geometry and tunable thresholds for the fixed road camera.

There is no `samples/camera.md` in this year's kit (dropped from an earlier
version of the task spec). `SCENE` below is a FIRST-PASS, EYEBALLED geometry
read manually off frames in `samples/frames/C3896/` (see
`src/extract_frames.py` / `src/verify_scene.py`) — coordinates are estimates,
not measured/verified. Re-check with `src/verify_scene.py`'s overlay image
before trusting this for anything beyond a rough sanity check; expect
corrected coordinates to replace these.

Lane/crossing/stop-line coordinates are stored NORMALIZED to [0, 1] (fraction
of frame width/height) since that's what's easiest to eyeball off a still
frame at any resolution. `SceneConfig.__post_init__` converts them to pixel
coordinates (matching the (cx, cy) trajectory points produced by
src/detect.py) using `width`/`height`, so every rule module and geometry
helper still operates in pixel space and never has to know about the
normalized/pixel distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.rules.geometry import distance, point_in_polygon, point_segment_distance, side_of_line


@dataclass
class LaneDirection:
    name: str
    polygon: list[tuple[float, float]]           # normalized [0,1] coords; converted to pixels in __post_init__
    direction: tuple[float, float]                # unit vector of legal travel direction (not scaled)
    disallowed_turns: list[str] = field(default_factory=list)  # e.g. ["left"] for illegal_turn
    allow_turn: bool = False  # True = this is a turn-permitted lane/pocket (e.g. a dedicated left/right
                               # turn lane, or the swept area a car legally occupies mid-turn). A vehicle
                               # heading a very different direction than `direction` while inside such a
                               # lane is not suspicious -- it's turning where it's allowed to. Read by
                               # src/rules/wrong_way.py and src/visualize_trails.py's
                               # classify_vehicle_behavior() to suppress wrong_way/turning flags there.
                               # NOT the same thing as `disallowed_turns` above: that's a per-ORIGIN-lane
                               # restriction ("turning left FROM this lane is illegal"); this is a
                               # per-DESTINATION/turn-pocket permission ("turning happens legally in this
                               # lane, don't flag the heading change"). The two can coexist on different
                               # lanes without conflicting.

    @property
    def id(self) -> str:
        return self.name


@dataclass
class CrossingZone:
    """Pedestrian crossing (zebra crossing) polygon."""
    name: str
    polygon: list[tuple[float, float]]            # normalized [0,1] coords

    @property
    def id(self) -> str:
        return self.name


@dataclass
class StopLine:
    name: str
    p1: tuple[float, float]                        # normalized [0,1] coords
    p2: tuple[float, float]
    approach_side: int = 1  # sign convention: points where side_of_line(pt, p1, p2)
                             # has this sign are still "before" (approaching) the
                             # line; the opposite sign means "past" it. Flip this
                             # ±1 if src/verify_scene.py's overlay shows it backwards.
    lane_name: str | None = None  # optional: which LaneDirection.name this stop line
                                    # belongs to, for scenes with multiple approaches/
                                    # signal phases. None is fine when there's only one
                                    # stop line in the scene (src/rules/red_light.py
                                    # doesn't require it -- it matches each stop line to
                                    # its nearest TrafficLightROI by distance instead).

    @property
    def id(self) -> str:
        return self.name


@dataclass
class SolidLine:
    """A solid lane-marking segment; crossing it is a `solid_line_crossing` violation."""
    id: str
    p1: tuple[float, float]                        # normalized [0,1] coords
    p2: tuple[float, float]


@dataclass
class TrafficLightROI:
    """A small box around a traffic-light lamp housing, cropped and
    classified by src/rules/traffic_light.py's classify_light_state() (HSV
    hue thresholding, no learned model). Used by src/rules/red_light.py to
    tell whether the signal was actually red when a vehicle crossed its
    nearest StopLine -- replacing the earlier "queue of >=2 stopped
    vehicles" proxy heuristic now that real color classification exists."""
    name: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0,1]; converted to pixels in __post_init__

    @property
    def id(self) -> str:
        return self.name


@dataclass
class EntryExitZone:
    """A small polygon marking where a road enters or exits the visible
    frame -- NOT a full lane, just a thin slice near a frame edge. Used by
    src/rules/zone_matching.py's whole-trajectory entry->exit classification,
    an alternative to LaneDirection's frame-by-frame heading-vs-direction
    comparison (see src/rules/wrong_way.py). Additive: does not replace or
    modify the lanes system, purely a parallel experiment living on its own
    SceneConfig.entry_exit_zones list."""
    name: str
    polygon: list[tuple[float, float]]            # normalized [0,1] coords; converted to pixels in __post_init__
    compass: str                                   # "north" | "east" | "south" | "west" -- which edge of the
                                                     # intersection this zone sits on; zone_matching.py's turn-type
                                                     # mapping (straight/left/right/wrong_way) currently only
                                                     # understands these 4 cardinal labels
    zone_type: str                                  # "entry" or "exit"

    @property
    def id(self) -> str:
        return self.name


@dataclass
class SceneConfig:
    fps: float = 25.0
    width: int = 0
    height: int = 0
    signal_visible: bool = False   # True if a traffic-light head is visible in frame.
                                    # stop_line.py still uses its own queue-based proxy
                                    # (unrelated to this flag); src/rules/red_light.py
                                    # uses real color classification via `traffic_lights`
                                    # + `light_history` below instead of this flag.

    lanes: list[LaneDirection] = field(default_factory=list)
    crossings: list[CrossingZone] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    solid_lines: list[SolidLine] = field(default_factory=list)
    entry_exit_zones: list[EntryExitZone] = field(default_factory=list)  # additive; see EntryExitZone's docstring
    traffic_lights: list[TrafficLightROI] = field(default_factory=list)
    # {roi_name: [(t_sec, "red"|"yellow"|"green"|"unknown"), ...]}, chronological.
    # Empty by default -- populated by src.rules.traffic_light.LightHistory
    # (solution.py does this once, before run_all_rules) since classify_light_state()
    # needs actual video frames, which the rules pipeline otherwise never sees (rules
    # only get `trajectories` + `scene`). Rides on `scene` rather than changing every
    # rule's detect(trajectories, scene) signature. See src/rules/red_light.py.
    light_history: dict = field(default_factory=dict)

    # Tunable thresholds. Units: px/s for speeds (pixel-space; camera isn't
    # calibrated to real-world meters, so all speed thresholds are relative to
    # this scene's own pixel scale — re-tune per video).
    thresholds: dict = field(default_factory=lambda: {
        "stopped_speed_px_s": 5.0,          # below this = "not moving"
        "stopped_vehicle_min_sec": 10.0,    # stopped_vehicle class definition
        "queue_stop_line_dist_px": 60.0,    # "waiting at the stop line" radius
        "congestion_speed_px_s": 15.0,    # tuned on dev labels (was 8)
        "congestion_min_vehicles": 12,    # tuned: 4 fired on every red-light queue
        "congestion_min_sec": 5.0,
        "jaywalk_min_sec": 3.0,            # tuned (was 1): brief kerb steps are noise
        "u_turn_window_sec": 4.0,
        "u_turn_min_angle_deg": 150.0,
        "wrong_way_min_angle_deg": 150.0,
        "wrong_way_min_sec": 1.0,
        "near_miss_decel_px_s2": 40.0,      # sudden deceleration threshold
        "near_miss_converge_px_s": 15.0,    # min closing speed to count as "converging"
        "near_miss_contact_px": 25.0,       # below this = treated as contact, not a near-miss
        "accident_closing_speed_px_s": 30.0,
        "accident_contact_px": 30.0,
        "accident_stall_min_sec": 1.5,
        "obstacle_min_sec": 3.0,
        "solid_line_margin_px": 5.0,
    })

    def __post_init__(self) -> None:
        """Scale normalized [0,1] geometry to pixel coordinates using width/height.
        No-op for an empty scene (width=height=0, no lanes/crossings/lines)."""
        def to_px(pt: tuple[float, float]) -> tuple[float, float]:
            return (pt[0] * self.width, pt[1] * self.height)

        for lane in self.lanes:
            lane.polygon = [to_px(p) for p in lane.polygon]
        for c in self.crossings:
            c.polygon = [to_px(p) for p in c.polygon]
        for sl in self.stop_lines:
            sl.p1 = to_px(sl.p1)
            sl.p2 = to_px(sl.p2)
        for sol in self.solid_lines:
            sol.p1 = to_px(sol.p1)
            sol.p2 = to_px(sol.p2)
        for z in self.entry_exit_zones:
            z.polygon = [to_px(p) for p in z.polygon]
        for tl in self.traffic_lights:
            x1, y1, x2, y2 = tl.box
            (px1, py1), (px2, py2) = to_px((x1, y1)), to_px((x2, y2))
            tl.box = (px1, py1, px2, py2)

    # ---- convenience geometry helpers, used by src/rules/*.py -------------
    def point_in_any_lane(self, pt: tuple[float, float]) -> LaneDirection | None:
        for lane in self.lanes:
            if point_in_polygon(pt, lane.polygon):
                return lane
        return None

    def point_in_any_crossing(self, pt: tuple[float, float]) -> CrossingZone | None:
        for c in self.crossings:
            if point_in_polygon(pt, c.polygon):
                return c
        return None

    def crossing_at(self, pt: tuple[float, float]) -> CrossingZone | None:
        return self.point_in_any_crossing(pt)

    def nearest_stop_line(self, pt: tuple[float, float]) -> tuple[StopLine, float] | None:
        best = None
        for sl in self.stop_lines:
            d = point_segment_distance(pt, sl.p1, sl.p2)
            if best is None or d < best[1]:
                best = (sl, d)
        return best

    def is_past_stop_line(self, pt: tuple[float, float], sl: StopLine) -> bool:
        """True if `pt` is on the opposite side of the p1->p2 line from `sl.approach_side`."""
        s = side_of_line(pt, sl.p1, sl.p2)
        if s == 0:
            return False
        sign = 1 if s > 0 else -1
        return sign != sl.approach_side

    def zone_at(self, pt: tuple[float, float]) -> EntryExitZone | None:
        """Any entry_exit_zones match (entry or exit type) -- used by
        src/rules/zone_matching.py. Additive/experimental, see EntryExitZone."""
        for z in self.entry_exit_zones:
            if point_in_polygon(pt, z.polygon):
                return z
        return None

    def nearest_traffic_light(self, sl: StopLine) -> TrafficLightROI | None:
        """The TrafficLightROI whose box center is closest to `sl`'s
        midpoint -- used by src/rules/red_light.py to pick which light
        governs a given stop line when there's no explicit link between
        them. Fine for a single-approach scene (one light, one stop line);
        revisit with an explicit name-based link if a scene ever needs
        multiple independent approaches/phases."""
        if not self.traffic_lights:
            return None
        mid = ((sl.p1[0] + sl.p2[0]) / 2, (sl.p1[1] + sl.p2[1]) / 2)
        best = None
        for tl in self.traffic_lights:
            x1, y1, x2, y2 = tl.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            d = distance(mid, center)
            if best is None or d < best[1]:
                best = (tl, d)
        return best[0] if best else None


# ---------------------------------------------------------------------------
# SCENE — hand-picked lanes + crossings, replacing the earlier single
# near_side_carriageway placeholder and the earlier 4-crosswalk set. All
# geometry below was picked with src/pick_points.py against
# samples/frames/C3896/clean_frame.jpg and visually verified by hand before
# being pasted in. lane_turn_left / lane_turn_right are dedicated turn
# lanes/pockets (allow_turn=True); lane_go_forward is a normal through-lane
# (allow_turn=False, default) so wrong_way.py's check still applies there
# at full strength. "crosswalk_2" below had a typo ("crosswlak_2") in the
# pasted source, corrected here (name only, coordinates unchanged).
SCENE = SceneConfig(
    fps=29.97,
    width=3840,
    height=2160,
    signal_visible=True,

    lanes=[
        LaneDirection(name="forward_move_line", polygon=[(0.351, 0.442), (0.285, 0.456), (0.229, 0.419), (0.207, 0.398), (0.178, 0.366), (0.162, 0.348), (0.156, 0.332), (0.156, 0.329), (0.176, 0.303), (0.178, 0.304), (0.182, 0.304), (0.183, 0.305), (0.233, 0.322), (0.234, 0.323)], direction=(0.894, 0.447)),
        LaneDirection(name="forward_move_line_2", polygon=[(0.417, 0.432), (0.351, 0.443), (0.308, 0.427), (0.304, 0.424), (0.298, 0.42), (0.294, 0.417), (0.291, 0.414), (0.28, 0.405), (0.269, 0.396), (0.267, 0.393), (0.233, 0.341), (0.233, 0.34), (0.233, 0.322), (0.233, 0.322), (0.311, 0.354), (0.392, 0.411)], direction=(0.927, 0.375)),
        LaneDirection(name="forward_move_line_3", polygon=[(0.284, 0.456), (0.216, 0.47), (0.212, 0.467), (0.142, 0.41), (0.142, 0.41), (0.14, 0.407), (0.14, 0.404), (0.14, 0.403), (0.156, 0.329), (0.247, 0.418)], direction=(0.877, 0.481)),
        LaneDirection(name="main_road1_moving_left_main_line", polygon=[(0.499, 0.806), (0.67, 0.996), (0.46, 0.999), (0.114, 0.992), (0.079, 0.868), (0.219, 0.851), (0.385, 0.829), (0.497, 0.806)], direction=(-0.965, 0.263)),
        LaneDirection(name="turn_left_line", polygon=[(0.317, 0.624), (0.291, 0.669), (0.264, 0.703), (0.172, 0.782), (0.113, 0.822), (0.064, 0.852), (0.025, 0.77), (0.102, 0.432), (0.233, 0.322), (0.297, 0.287), (0.298, 0.287), (0.301, 0.29)], direction=(-0.382, 0.924), allow_turn=True),
        LaneDirection(name="turn_right_line", polygon=[(0.609, 0.509), (0.608, 0.512), (0.519, 0.532), (0.361, 0.391), (0.297, 0.287), (0.376, 0.273), (0.382, 0.275)], direction=(0.878, 0.479), allow_turn=True),
    ],
    crossings=[
        CrossingZone(name="crosswalk_main_road2", polygon=[(0.674, 0.506), (0.622, 0.473), (0.901, 0.437), (0.952, 0.461)]),
        CrossingZone(name="crosswalk_main_road1", polygon=[(0.174, 0.565), (0.2, 0.603), (0.61, 0.512), (0.56, 0.481)]),
        CrossingZone(name="crossing2_main_road1", polygon=[(0.094, 0.701), (0.159, 0.641), (0.37, 0.829), (0.258, 0.86)]),
        CrossingZone(name="crossing3_main_road1", polygon=[(0.259, 0.861), (0.371, 0.829), (0.471, 0.999), (0.325, 0.998)]),
    ],
    stop_lines=[
        # approach_side=-1: side_of_line() is negative for vehicles still upstream
        # (top-left) of this p1->p2 line; +1 had it backwards.
        StopLine(name="stop_line_main_road1", p1=(0.151, 0.487), p2=(0.486, 0.421), approach_side=-1, lane_name=None),
    ],

    # *** PLACEHOLDER GEOMETRY -- NOT HAND-PICKED, ROUGH GUESSES ONLY ***
    # Thin border-strip polygons near each frame edge, just to exercise
    # src/rules/zone_matching.py end-to-end. Not verified against the actual
    # frame content (unlike everything else in this SCENE, which was
    # eyeballed/picked against samples/frames/C3896/clean_frame.jpg). Re-pick
    # real ones with src/pick_points.py before trusting zone_matching.py's
    # output for anything beyond "does the logic run".
    entry_exit_zones=[
        EntryExitZone(name="entry_north", polygon=[(0.30, 0.0), (0.50, 0.0), (0.50, 0.05), (0.30, 0.05)], compass="north", zone_type="entry"),
        EntryExitZone(name="exit_north", polygon=[(0.50, 0.0), (0.70, 0.0), (0.70, 0.05), (0.50, 0.05)], compass="north", zone_type="exit"),
        EntryExitZone(name="entry_south", polygon=[(0.30, 0.95), (0.50, 0.95), (0.50, 1.0), (0.30, 1.0)], compass="south", zone_type="entry"),
        EntryExitZone(name="exit_south", polygon=[(0.50, 0.95), (0.70, 0.95), (0.70, 1.0), (0.50, 1.0)], compass="south", zone_type="exit"),
        EntryExitZone(name="entry_west", polygon=[(0.0, 0.30), (0.05, 0.30), (0.05, 0.50), (0.0, 0.50)], compass="west", zone_type="entry"),
        EntryExitZone(name="exit_west", polygon=[(0.0, 0.50), (0.05, 0.50), (0.05, 0.70), (0.0, 0.70)], compass="west", zone_type="exit"),
        EntryExitZone(name="entry_east", polygon=[(0.95, 0.30), (1.0, 0.30), (1.0, 0.50), (0.95, 0.50)], compass="east", zone_type="entry"),
        EntryExitZone(name="exit_east", polygon=[(0.95, 0.50), (1.0, 0.50), (1.0, 0.70), (0.95, 0.70)], compass="east", zone_type="exit"),
    ],
    traffic_lights=[
        # Box tightened around just the lit lamp COLUMN (not the full 3-lamp
        # housing) to raise color-fraction margins: the housing is obliquely
        # tilted, so the red lamp sits top-right and the green lamp bottom-left
        # -- this box spans both while excluding neighbouring dark housing and
        # the reflection just past the lamps (which used to produce a spurious
        # 'yellow'). Measured across the full clip vs the old wider box: red
        # peak 0.028->0.105, green 0.032->0.069, no red/green frame regressed
        # (one prior 'yellow' misread became correct 'green'). Was
        # box=(0.596, 0.337, 0.61, 0.389).
        TrafficLightROI(name="traffic_light", box=(0.6003, 0.3407, 0.6052, 0.3796)),
    ],
)


# ---------------------------------------------------------------------------
# SCENE_C3905 — the camera is re-framed between clips, so C3896's geometry
# above does not line up on C3905. Traced against a median-of-61-frames
# background plate of C3905 (vehicles removed): lane dividers, stop line and
# crosswalks fitted to the segmented white paint (Hough + morphology), lane
# directions checked against YOLO+ByteTrack trajectories over the whole clip.
# approach_lane_1..5 = curb lane .. median lane of the approach that stops at
# stop_line_approach; far_carriageway = opposing traffic above the median;
# junction = everything past the stop line (straight-through, left turns,
# U-turns and the right-turn slip lane all share it, hence allow_turn=True).
SCENE_C3905 = SceneConfig(
    fps=29.97,
    width=3840,
    height=2160,
    signal_visible=True,

    lanes=[
        LaneDirection(name="approach_lane_1", polygon=[(0.0391, 0.3085), (0.2161, 0.4949), (0.1477, 0.5102), (0.0391, 0.3657)], direction=(0.838, 0.546)),
        LaneDirection(name="approach_lane_2", polygon=[(0.0391, 0.2516), (0.2901, 0.4782), (0.2161, 0.4949), (0.0391, 0.3085)], direction=(0.879, 0.477)),
        LaneDirection(name="approach_lane_3", polygon=[(0.0391, 0.2026), (0.3573, 0.4630), (0.2901, 0.4782), (0.0391, 0.2516)], direction=(0.901, 0.434)),
        LaneDirection(name="approach_lane_4", polygon=[(0.0391, 0.1664), (0.4255, 0.4477), (0.3573, 0.4630), (0.0391, 0.2026)], direction=(0.918, 0.397)),
        LaneDirection(name="approach_lane_5", polygon=[(0.0391, 0.1240), (0.4870, 0.4338), (0.4255, 0.4477), (0.0391, 0.1664)], direction=(0.929, 0.370)),
        LaneDirection(name="far_carriageway", polygon=[(0.0391, 0.0440), (0.2797, 0.1269), (0.6380, 0.2593), (0.7031, 0.3056), (0.8594, 0.4028), (0.9635, 0.4329), (1.0000, 0.4444), (1.0000, 0.6620), (0.6875, 0.5255), (0.6250, 0.4861), (0.6164, 0.4716), (0.0391, 0.1257)], direction=(-0.945, -0.327)),
        LaneDirection(name="junction", polygon=[(0.1477, 0.5102), (0.4870, 0.4338), (0.6250, 0.4861), (0.6875, 0.5255), (1.0000, 0.6620), (1.0000, 1.0000), (0.0000, 1.0000), (0.0000, 0.7870), (0.0990, 0.6944), (0.1510, 0.6648), (0.1745, 0.6250)], direction=(0.921, 0.389), allow_turn=True),
    ],
    crossings=[
        CrossingZone(name="crosswalk_main", polygon=[(0.1620, 0.5843), (0.5891, 0.4861), (0.6094, 0.5176), (0.1732, 0.6241)]),
        CrossingZone(name="crosswalk_far_side", polygon=[(0.6302, 0.4722), (0.9557, 0.4306), (0.9635, 0.4560), (0.6406, 0.5116)]),
        CrossingZone(name="crosswalk_diagonal", polygon=[(0.0961, 0.6819), (0.1523, 0.6676), (0.2737, 0.7171), (0.4049, 0.8847), (0.4661, 0.9995), (0.3208, 0.9995), (0.1945, 0.8759)]),
    ],
    stop_lines=[
        StopLine(name="stop_line_approach", p1=(0.1477, 0.5102), p2=(0.4870, 0.4338), approach_side=-1),
    ],
    traffic_lights=[
        # Front (camera-facing) head on the median-nose pole; verified
        # red/green cycle on C3905 with classify_light_state().
        TrafficLightROI(name="traffic_light", box=(0.6036, 0.3380, 0.6107, 0.3843)),
    ],
)

# Per-video geometry, keyed by file stem. Videos not listed fall back to SCENE
# (C3896's framing) -- add an entry here when a clip's framing differs.
SCENES: dict[str, SceneConfig] = {
    "C3896": SCENE,
    "C3905": SCENE_C3905,
}


def scene_for(video_path: str) -> SceneConfig:
    from pathlib import Path
    return SCENES.get(Path(video_path).stem.upper(), SCENE)
