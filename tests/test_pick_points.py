"""tests/test_pick_points.py — automated checks for src/pick_points.py's
click-collection state machine and, critically, the threaded_input() freeze
fix — all run without opening a real cv2 window or GUI display.

`collect_clicks()` and `threaded_input()` are deliberately written to take
plain callables (key source, pump function, add/undo/reset/len) instead of
calling cv2.waitKey()/input() directly, so they can be driven here with
scripted fakes instead of a real mouse/keyboard/terminal.

Run: python tests/test_pick_points.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pick_points import PointPicker, _parse_direction, collect_clicks, collect_polygon_clicks, threaded_input


def test_point_picker_basics() -> None:
    picker = PointPicker(1280, 720)
    for pt in [(100, 300), (600, 300), (600, 500), (100, 500)]:
        picker.add_point(*pt)
    picker.finish_shape("lane_a", "lane", direction=(1.0, 0.0))
    assert len(picker.shapes) == 1

    picker.add_point(10, 10)
    picker.undo()
    assert picker.current == []

    for pt in [(330, 580), (490, 580), (490, 680), (330, 680)]:
        picker.add_point(*pt)
    picker.finish_shape("crossing_a", "crossing")

    picker.add_point(330, 560)
    picker.add_point(490, 560)
    picker.finish_shape("stop_a", "stop_line")

    assert len(picker.list_shapes()) == 3
    removed = picker.delete_shape(1)
    assert removed.name == "crossing_a"
    assert len(picker.list_shapes()) == 2

    code = picker.to_python_code(source="unit_test.jpg")
    assert "crossing_a" not in code, "deleted shape leaked into output"
    assert "lane_a" in code and "stop_a" in code
    print("test_point_picker_basics: PASS")


def test_collect_clicks_success() -> None:
    """Clicks delivered asynchronously via pump_fn, mirroring the real
    on_mouse callback firing a click during a redraw tick."""
    pts: list[tuple[float, float]] = []
    scripted = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    state = {"i": 0}

    def pump_fn() -> None:
        if state["i"] < len(scripted):
            pts.append(scripted[state["i"]])
            state["i"] += 1

    ok = collect_clicks(4, pts.append, lambda: pts.pop() if pts else None, pts.clear,
                         lambda: len(pts), key_fn=lambda: "", pump_fn=pump_fn)
    assert ok
    assert pts == scripted
    print("test_collect_clicks_success: PASS")


def test_collect_clicks_undo_and_reset() -> None:
    pts: list[tuple[float, float]] = []
    script = [
        ("", (0.0, 0.0)),
        ("", (1.0, 0.0)),
        ("", (2.0, 0.0)),
        ("u", None),          # -> back to len 2
        ("", (3.0, 0.0)),      # -> len 3
        ("r", None),           # -> len 0
        ("", (4.0, 0.0)),
        ("", (5.0, 0.0)),
        ("", (6.0, 0.0)),
        ("", (7.0, 0.0)),      # -> len 4, loop exits
    ]
    it = iter(script)
    shared: dict = {"click": None}

    def key_fn() -> str:
        key, click = next(it)
        shared["click"] = click
        return key

    def pump_fn() -> None:
        if shared["click"] is not None:
            pts.append(shared["click"])
            shared["click"] = None

    ok = collect_clicks(4, pts.append, lambda: pts.pop() if pts else None, pts.clear,
                         lambda: len(pts), key_fn, pump_fn)
    assert ok
    assert pts == [(4.0, 0.0), (5.0, 0.0), (6.0, 0.0), (7.0, 0.0)], f"unexpected final points: {pts}"
    print("test_collect_clicks_undo_and_reset: PASS")


def _run_scripted_polygon(add_fn, undo_fn, reset_fn, len_fn, script, **kwargs) -> bool:
    """Drive collect_polygon_clicks with a fixed (key, click_or_None) script,
    one entry consumed per loop iteration — mirrors how the real on_mouse
    callback and cv2.waitKey() interleave, but deterministic and instant."""
    it = iter(script)
    shared: dict = {"click": None}

    def key_fn() -> str:
        key, click = next(it)
        shared["click"] = click
        return key

    def pump_fn() -> None:
        if shared["click"] is not None:
            add_fn(shared["click"])
            shared["click"] = None

    return collect_polygon_clicks(add_fn, undo_fn, reset_fn, len_fn, key_fn, pump_fn, **kwargs)


def test_collect_polygon_clicks_variable_length() -> None:
    """The actual feature request: a lane polygon with 6 or 8 points, not a
    fixed 4, finishing on 'd' or Enter ('\\r')."""
    for n, done_key in ((6, "d"), (8, "\r")):
        pts: list[tuple[float, float]] = []
        script = [("", (float(i), float(i))) for i in range(n)] + [(done_key, None)]
        ok = _run_scripted_polygon(pts.append, lambda: pts.pop() if pts else None, pts.clear,
                                    lambda: len(pts), script)
        assert ok, f"n={n} done_key={done_key!r} should succeed"
        assert len(pts) == n, f"expected {n} points, got {len(pts)}: {pts}"
    print("test_collect_polygon_clicks_variable_length: PASS (6 pts via 'd', 8 pts via Enter)")


def test_collect_polygon_clicks_enforces_min_points() -> None:
    """Pressing 'd' with fewer than min_points must NOT finish the shape."""
    pts: list[tuple[float, float]] = []
    script = [
        ("", (0.0, 0.0)),
        ("", (1.0, 0.0)),
        ("d", None),          # only 2 points < min 3 -> must be rejected, loop continues
        ("", (2.0, 0.0)),
        ("d", None),          # now 3 points -> finishes
    ]
    ok = _run_scripted_polygon(pts.append, lambda: pts.pop() if pts else None, pts.clear,
                                lambda: len(pts), script, min_points=3)
    assert ok
    assert len(pts) == 3, f"premature 'd' at 2 points should have been rejected, got {pts}"
    print("test_collect_polygon_clicks_enforces_min_points: PASS")


def test_collect_polygon_clicks_undo_and_reset() -> None:
    pts: list[tuple[float, float]] = []
    script = [
        ("", (0.0, 0.0)),
        ("", (1.0, 0.0)),
        ("", (2.0, 0.0)),
        ("u", None),           # -> back to 2
        ("", (3.0, 0.0)),      # -> 3
        ("r", None),           # -> 0
        ("", (4.0, 0.0)),
        ("", (5.0, 0.0)),
        ("", (6.0, 0.0)),
        ("", (7.0, 0.0)),
        ("", (8.0, 0.0)),      # -> 5 points (well above min 3)
        ("d", None),           # finish
    ]
    ok = _run_scripted_polygon(pts.append, lambda: pts.pop() if pts else None, pts.clear,
                                lambda: len(pts), script)
    assert ok
    assert pts == [(4.0, 0.0), (5.0, 0.0), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0)], f"unexpected final points: {pts}"
    print("test_collect_polygon_clicks_undo_and_reset: PASS")


def test_collect_polygon_clicks_quit_aborts_and_resets() -> None:
    pts: list[tuple[float, float]] = [(1.0, 1.0), (2.0, 2.0)]  # points already placed

    ok = collect_polygon_clicks(pts.append, lambda: pts.pop() if pts else None, pts.clear,
                                 lambda: len(pts), key_fn=lambda: "q", pump_fn=lambda: None)
    assert ok is False
    assert pts == [], "abort ('q') must reset the pending points"
    print("test_collect_polygon_clicks_quit_aborts_and_resets: PASS")


def test_lane_code_output_with_more_than_4_points() -> None:
    """End-to-end through PointPicker: an 8-point lane polygon must still
    print correctly, and crossing/stop_line printing must be unaffected."""
    picker = PointPicker(1280, 720)
    octagon = [(100 + i * 10, 300 + i * 5) for i in range(8)]
    for pt in octagon:
        picker.add_point(*pt)
    picker.finish_shape("curvy_lane", "lane", direction=(0.6, 0.8))

    for pt in [(330, 580), (490, 580), (490, 680), (330, 680)]:
        picker.add_point(*pt)
    picker.finish_shape("crossing_a", "crossing")

    picker.add_point(330, 560)
    picker.add_point(490, 560)
    picker.finish_shape("stop_a", "stop_line")

    code = picker.to_python_code(source="octagon_test.jpg")
    assert 'name="curvy_lane"' in code

    lane_line = next(line for line in code.splitlines() if "curvy_lane" in line)
    polygon_str = lane_line.split("polygon=[")[1].split("], direction=")[0]
    printed_points = [p.strip() for p in polygon_str.split("), (")]
    assert len(printed_points) == 8, f"expected 8 coordinate tuples in the lane line, got {len(printed_points)}: {lane_line}"
    # round-trip check: the normalized coords must match the original clicks / img size
    first_x, first_y = octagon[0]
    expected_first = f"({round(first_x / 1280, 3)}, {round(first_y / 720, 3)}"
    assert polygon_str.startswith(expected_first), f"first point mismatch: {polygon_str[:30]!r} vs {expected_first!r}"

    assert 'name="crossing_a"' in code and "direction=" not in code.split("crossings=[")[1].split("],")[0]
    assert 'name="stop_a"' in code
    print("test_lane_code_output_with_more_than_4_points: PASS")


def test_collect_clicks_quit_aborts_and_resets() -> None:
    pts: list[tuple[float, float]] = [(1.0, 1.0)]  # one point already placed

    ok = collect_clicks(4, pts.append, lambda: pts.pop() if pts else None, pts.clear,
                         lambda: len(pts), key_fn=lambda: "q", pump_fn=lambda: None)
    assert ok is False
    assert pts == [], "abort ('q') must reset the pending points, not leave them dangling"
    print("test_collect_clicks_quit_aborts_and_resets: PASS")


def test_threaded_input_keeps_pumping_during_slow_input() -> None:
    """The actual freeze-fix check: proves the pump function (which, for
    real, is `redraw() + cv2.waitKey()`) keeps firing on the caller's thread
    WHILE input() is blocked on a background thread. Before this fix, the
    equivalent of pump_fn simply never ran for the duration of input() —
    that starvation is what got the cv2 window flagged unresponsive."""
    call_count = {"n": 0}

    def pump_fn() -> None:
        call_count["n"] += 1

    def fake_input(prompt: str = "") -> str:
        time.sleep(0.25)
        return "hello"

    with patch("builtins.input", fake_input):
        t0 = time.time()
        result = threaded_input("prompt: ", pump_fn, poll_sec=0.03)
        elapsed = time.time() - t0

    assert result == "hello"
    assert elapsed < 1.0, f"threaded_input took {elapsed:.2f}s to return after input() resolved — should be prompt"
    assert call_count["n"] >= 5, (
        f"pump_fn only fired {call_count['n']}x during a 0.25s input() block — "
        "the GUI thread would have been starved of cv2.waitKey(), which is the freeze bug"
    )
    print(f"test_threaded_input_keeps_pumping_during_slow_input: PASS (pumped {call_count['n']}x during 0.25s block)")


def test_threaded_input_handles_eof() -> None:
    def fake_input(prompt: str = "") -> str:
        raise EOFError()

    with patch("builtins.input", fake_input):
        result = threaded_input("prompt: ", lambda: None, poll_sec=0.03)
    assert result == "q", "EOF on stdin should be treated as a quit request, not hang forever"
    print("test_threaded_input_handles_eof: PASS")


def test_parse_direction() -> None:
    assert _parse_direction("1,0") == (1.0, 0.0)
    assert _parse_direction("0,-1") == (0.0, -1.0)
    assert _parse_direction("") is None
    assert _parse_direction("not a vector") is None
    print("test_parse_direction: PASS")


if __name__ == "__main__":
    test_point_picker_basics()
    test_collect_clicks_success()
    test_collect_clicks_undo_and_reset()
    test_collect_clicks_quit_aborts_and_resets()
    test_collect_polygon_clicks_variable_length()
    test_collect_polygon_clicks_enforces_min_points()
    test_collect_polygon_clicks_undo_and_reset()
    test_collect_polygon_clicks_quit_aborts_and_resets()
    test_lane_code_output_with_more_than_4_points()
    test_threaded_input_keeps_pumping_during_slow_input()
    test_threaded_input_handles_eof()
    test_parse_direction()
    print("\nALL PICK_POINTS TESTS PASSED")
