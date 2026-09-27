"""src/visualize_trails.py — live, fading, behavior-colored trajectory trails
per tracked vehicle, baked into the output video frame by frame (a "comet
trail" effect, not a single static overlay computed after the fact).

Standalone visualization/demo tool, same status as src/visualize_tracks.py:
reads src.detect.VideoTracker's TrackPoint output and src.config.SCENE
read-only. Does not modify solution.py, detect.py, postprocess.py, config.py,
or anything in src/rules/ — has no bearing on the actual submission pipeline
or its scoring.

Usage:
    python -m src.visualize_trails samples/C3896.MP4 --output-dir samples/frames/C3896 --max-frames 60 --device cpu
    # full video on Colab/Kaggle:
    python -m src.visualize_trails /content/C3896.MP4 --output-dir /content/out --device cuda
    # quick screen-only preview on your own machine, nothing saved to disk:
    python -m src.visualize_trails samples/C3896.MP4 --live
    # faster live preview (half-res detection/display, only detect every 2nd frame):
    python -m src.visualize_trails samples/C3896.MP4 --live --scale 0.4 --stride 2
    # live preview without the lane/crossing/stop_line overlay:
    python -m src.visualize_trails samples/C3896.MP4 --live --no-show-scene

Output (written into --output-dir; NOT produced with --live, see below):
    trails_output.mp4   annotated video: each vehicle gets a box + a fading
                         trail from wherever the track started, up to --trail-length points,
                         colored by classify_vehicle_behavior() below.

--live mode: opens a cv2 window and plays the video with the same live boxes
+ fading trails, but shows it on screen instead of writing trails_output.mp4
(or anything else) to disk. Runs to the end of the video, or until you press
'q' in the window — no --max-frames cap applies here (a value passed
alongside --live is ignored, with a note printed so you know). Needs a real
display; this is for eyeballing on your own machine, not something the
harness or a headless box can use.

--live-only flags (both are genuine speed/quality trade-offs, not free —
see run_live()'s docstring for exactly what each one costs):
    --scale F         resize frames to this fraction before detection AND
                       display (default 0.5) — the dominant CPU cost is
                       YOLO on a full 3840x2160 frame every frame; this cuts
                       that down at the cost of detection recall on small/
                       distant vehicles and box precision.
    --stride N        only run detection every Nth raw frame (shared with
                       the non-live path, default 1); in --live, frames in
                       between reuse the last known boxes/trails so the
                       video itself keeps playing smoothly, at the cost of
                       box positions lagging the real vehicle between
                       detections and a coarser trail.
    --show-scene / --no-show-scene
                       overlay scene.lanes/crossings/stop_lines as static
                       outlines (default ON) since the camera doesn't move.

Color legend (BGR, drawn in the video and printed in the legend at startup):
    green   "aligned"    heading matches its lane's expected direction
    orange  "turning"    heading has rotated >= --turn-threshold-deg over
                          the trail window (independent of lane direction)
    red     "wrong_way"  heading opposes its lane's expected direction
    gray    "neutral"    not inside any mapped lane, or near-stationary --
                          these are exactly the cases wrong_way.py itself
                          already excludes before flagging a violation

Pedestrians: boxed and labeled too (VideoTracker's default class filter
already includes "person" -- detection was never vehicle-only; they just
weren't being drawn). No trail, box+label only -- that alone keeps them
visually distinct from vehicles, on top of using a completely different
color family than the green/orange/red/gray vehicle-behavior palette. FOUR
states, not two -- an earlier version only checked "inside a crossing?" and
wrongly painted anyone off a crossing, INCLUDING someone standing still on a
sidewalk, as "jaywalking?". jaywalking.py's real rule (confirmed correct by
reading it, not touched here) requires being ON THE ROAD, not merely off a
crossing, AND sustained for jaywalk_min_sec -- classify_pedestrian_location()
now reuses both checks:
    cyan   "person ... [crossing]"     inside a mapped crossing polygon
    gray   "person ... [off-road]"     not on the carriageway at all
                                        (sidewalk/median/etc) -- never flagged
    amber  "person ... [on road?]"     on the road, outside a crossing, but
                                        not yet sustained for jaywalk_min_sec
    pink   "person ... [jaywalking?]"  on the road, outside a crossing,
                                        continuously, for >= jaywalk_min_sec
The "?" on the last one is still deliberate -- see
classify_pedestrian_location()'s docstring for why even this isn't literally
jaywalking.py's own event output.

Trail persistence: each track's trail is drawn from wherever that track ID
first appeared, continuously, for as long as it stays alive -- not a short
flickering window. It's capped at --trail-length points per track (default
400, see TRAIL_LENGTH_DEFAULT's comment for why) so a very long-lived track
doesn't grow unbounded; the oldest ~15% of a long trail is simply not drawn
at all (TRAIL_DROP_OLDEST_FRACTION), the rest is drawn at full opacity with
a thickness that tapers from TRAIL_THICKNESS_MAX near the vehicle down to
1px with age. A track's history is only dropped once that track hasn't been
seen for ~90 detection calls (PRUNE_AFTER_DETECTIONS) -- i.e. once it's
actually gone, not on a short display timer -- which bounds total memory
across a long video without ever cutting off a trail that's still being
actively tracked. Separately, classify_vehicle_behavior()'s own "recent
heading" window (CLASSIFY_WINDOW) is intentionally NOT tied to the display
trail length -- see classify_recent().

Rendering: draw_trail() draws directly onto the real frame with cv2.line(),
full opaque color, no separate overlay/alpha layer -- see its docstring for
why, after comparing against a known-working reference implementation
(MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking) turned up that its
"fade" is a thickness taper, not transparency, and that our earlier
alpha-blended version (mathematically correct, but a flat 1px line) was
invisible during real playback for a reason unrelated to the alpha math.

Dependencies: only cv2 + numpy, same as visualize_tracks.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from src.config import SCENE, SceneConfig, scene_for
from src.detect import PEDESTRIAN_CLASSES, VEHICLE_CLASSES, TrackPoint, VideoTracker
from src.rules.geometry import angle_between, ground_point, side_of_line, unit_vector
from src.rules.red_light import _crossing_within_segment
from src.rules.traffic_light import classify_light_state

TRAIL_LENGTH_DEFAULT = 400         # persistent trail cap per track, in DETECTED points (not raw frames).
                                    # Chosen at the upper end of the requested 300-500 range: at the
                                    # default stride=1 (~30fps), 400 points is ~13s of history -- the
                                    # full driven path for the vast majority of a vehicle's time in
                                    # frame on these clips, while still bounding per-track memory to a
                                    # fixed, small amount (400 tuples of 3 floats is trivial; the real
                                    # cost of a long trail is REDRAW time, not memory -- see draw_trail()).
CLASSIFY_WINDOW = 25               # UNCHANGED from the original (pre-persistence) trail length. Turn/
                                    # wrong-way classification deliberately keeps looking at only this
                                    # many RECENT points regardless of how long the persistent display
                                    # trail (TRAIL_LENGTH_DEFAULT) grows -- otherwise classify_vehicle_
                                    # behavior()'s "earliest vs current heading" comparison would silently
                                    # start comparing against a point from minutes ago instead of ~1s ago,
                                    # changing its actual behavior even though its code is untouched.
TRAIL_DROP_OLDEST_FRACTION = 0.15  # the OLDEST 15% of a track's trail is simply not drawn at all (a
                                    # hard cutoff); the remaining 85% (the bulk of the driven path) IS
                                    # drawn, at full opaque color -- no alpha fade, see draw_trail()
PRUNE_AFTER_DETECTIONS = 90        # a track's history is only dropped after ~90 detection-frames
                                    # (~3s at stride=1/30fps) of not appearing -- a generous grace period
                                    # well past a brief occlusion, so a trail is never cut off while the
                                    # vehicle is still actually being tracked; this just bounds total
                                    # memory across a long video (many distinct track IDs over minutes),
                                    # separate from the per-track TRAIL_LENGTH_DEFAULT cap above.
TURN_THRESHOLD_DEG_DEFAULT = 15.0  # Signal 2 boundary for classify_vehicle_behavior(): heading-change
                                    # RATE (angular velocity) at/above this over the window -> "turning".
                                    # Also the aligned/turning split for that signal (aligned requires
                                    # heading_delta BELOW this). CLI-tunable via --turn-threshold-deg.
                                    # Was 35.0 under the old single-signal "earliest-vs-current" scheme;
                                    # 15.0 suits the new dedicated angular-velocity signal (see the
                                    # dual-signal constants just below and classify_vehicle_behavior()).

# --- classify_vehicle_behavior() dual-signal thresholds/windows -------------------------
# DISPLAY-ONLY tuning knobs for the live visualizer's per-track label. src/rules/wrong_way.py's
# scored rule keeps its OWN thresholds (scene.thresholds["wrong_way_min_*"]) and is untouched.
# The three named decision thresholds are: VIZ_ALIGNED_MAX_DEG (Signal 1, angle vs lane dir),
# TURN_THRESHOLD_DEG_DEFAULT above (Signal 2, heading-change rate), and, for wrong_way, the
# scored rule's own scene.thresholds["wrong_way_min_angle_deg"] (reused, not duplicated here).
VIZ_ALIGNED_MAX_DEG = 20.0     # Signal 1: angle_dev (recent heading vs lane.direction) at/above this
                                #  -> at least "turning"; below it (and low heading-change rate) -> "aligned".
VIZ_HEADING_SEG_PTS = 8        # points per heading estimate (~0.27s @30fps): a heading vector is the
                                #  displacement between two samples this many points apart, NOT a single
                                #  frame pair -- averages out per-frame detection jitter (Signal 1 & 2 both).
VIZ_ANGVEL_WINDOW_PTS = 15     # points (~0.5s @30fps) back to the EARLIER heading that the recent heading
                                #  is compared against for the heading-change-rate signal (Signal 2).

# --- new movement sub-categories (additive; layered on top of the dual-signal logic) -----
VIZ_TURN_CROSS_EPSILON = 0.02        # |cross(unit earlier_heading, unit recent_heading)| below this =
                                      #  ambiguous rotation direction -> keep the generic "turning" label
                                      #  (cross of unit vectors == sin(turn angle), so 0.02 ~= 1.1 deg).
VIZ_LANE_CHANGE_WINDOW_PTS = 30      # a lane change = current lane polygon differs from the one this many
                                      #  detections ago, while heading stays lane-aligned (angle_dev low).
VIZ_LANE_ADJACENT_MAX_ANGLE_DEG = 30.0  # only call it a lane change if old & new lanes point roughly the
                                      #  same way (angle_between(old.direction,new.direction) < this) -- else
                                      #  it's a real turn between differently-directed lanes, not a lane change.
U_TURN_MIN_ANGLE_DEG = 150.0         # recent heading vs the track's ORIGINAL (lane-entry) heading at/above
                                      #  this = a near-full reversal.
U_TURN_MIN_SUSTAIN_FRAMES = 20       # ...sustained this many consecutive detections -> "u_turn" (distinguishes
                                      #  a deliberate gradual reversal from a tracker glitch/ID switch, and from
                                      #  wrong_way, which enters already-opposed so never diverges from its own
                                      #  original heading). Checked BEFORE wrong_way -- see classify_vehicle_behavior.
VIZ_LANE_MEMORY_FRAMES = 45          # "lane memory": when point_in_any_lane() returns None (e.g. a vehicle out in
                                      #  the open intersection CENTER, which has no lane polygon by design), keep
                                      #  using the track's last-known lane's direction for up to this many
                                      #  consecutive detections (~1.5s @30fps) instead of dropping to "neutral" --
                                      #  so a car mid-turn through the gap still reads turning_left/right. Beyond
                                      #  this gap the memory is considered stale and it falls back to "neutral".
PED_HISTORY_LENGTH = 90            # ~3s at stride=1/30fps -- only needs to cover jaywalk_min_sec (1.0s
                                    # by default) of lookback for classify_pedestrian_location()'s
                                    # sustained-duration check, with margin. No trail is drawn from this
                                    # (pedestrians don't get a comet trail) -- it exists purely so that
                                    # check can walk backward through recent positions.
TRAIL_THICKNESS_MAX = 6            # peak thickness (px) at the newest trail segment, tapering toward
                                    # 1px with age -- see draw_trail()'s docstring. NOT flat 1px: real
                                    # pixel evidence from an actual trails_output.mp4 showed a constant
                                    # 1px line is invisible at 4K/normal playback scale even though the
                                    # underlying pixels were correct; a known-working reference
                                    # (YOLOv8-DeepSORT-Object-Tracking) uses up to 8px near the vehicle.
                                    # 6 is a deliberate compromise: thick enough to actually read during
                                    # playback, thinner than the reference's 8px "not a highlighter stroke".
TRAIL_TAPER_LENGTH = 40             # thickness ramps LINEARLY from TRAIL_THICKNESS_MAX down to 1px over
                                    # this many most-recent points, then stays flat 1px beyond that. A
                                    # first attempt copied the reference's inverse-sqrt falloff shape
                                    # directly and, measured on a real 150-point track, only made 2/128
                                    # segments thickness>=4 -- an easy-to-miss sliver against a much
                                    # longer trail than the reference's own 64-point buffer. A wider,
                                    # linear taper (measured: 21/128 segments >=4 on the same real
                                    # track) is a fixed absolute size regardless of how long the overall
                                    # persistent trail (TRAIL_LENGTH_DEFAULT) has grown.
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)

COLOR_ALIGNED = (0, 200, 0)      # BGR: green
COLOR_TURNING = (0, 140, 255)    # BGR: orange
COLOR_WRONG_WAY = (0, 0, 255)    # BGR: red
COLOR_NEUTRAL = (150, 150, 150)  # BGR: gray

BEHAVIOR_COLORS = {
    "aligned": COLOR_ALIGNED,
    "turning": COLOR_TURNING,
    "wrong_way": COLOR_WRONG_WAY,
    "neutral": COLOR_NEUTRAL,
    # new movement sub-categories (see classify_vehicle_behavior). turning_left/right are
    # shades of the orange "turning" family; lane changes a distinct cyan/teal; u_turn magenta.
    "turning_left": (0, 180, 255),      # BGR: orange-yellow
    "turning_right": (0, 100, 200),     # BGR: darker orange
    "lane_change_left": (200, 200, 0),  # BGR: cyan
    "lane_change_right": (200, 120, 0), # BGR: teal-blue
    "u_turn": (200, 0, 200),            # BGR: magenta
}

# Red-light violation overlay (display-only; mirrors the LIVE crossing-on-red
# check in run_live(), which reuses red_light.py's side_of_line crossing logic
# but does NOT touch that scored rule). Drawn OVER the normal behavior label so
# a flagged car is unmistakable. Bright red + thicker than a behavior box.
REDLIGHT_COLOR = (0, 0, 255)   # BGR: red
REDLIGHT_LABEL = "RED LIGHT"

# Scene-geometry overlay colors (--show-scene) -- deliberately NOT reusing any
# BEHAVIOR_COLORS value, so a static lane outline is never visually confused
# with a live vehicle's behavior-colored box/trail.
SCENE_LANE_COLOR = (255, 180, 0)      # BGR: blue
SCENE_CROSSING_COLOR = (255, 0, 255)  # BGR: magenta
SCENE_STOPLINE_COLOR = (0, 255, 255)  # BGR: yellow

# traffic_light_roi circle colors, keyed by src.rules.traffic_light.classify_light_state()'s
# possible return values -- 'unknown' (incl. "not classified yet") reads gray, same convention
# PED_OFF_ROAD_COLOR/COLOR_NEUTRAL already use elsewhere in this file for "nothing to report".
LIGHT_STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "green": (0, 200, 0),
    "unknown": (150, 150, 150),
}

# Pedestrian colors -- deliberately a completely different hue family from
# BEHAVIOR_COLORS (green/orange/red/gray) and from the scene-geometry colors
# above, so pedestrians read as a different CATEGORY at a glance, not just a
# different value within the vehicle-behavior system. No trail is drawn for
# pedestrians (box + label only) -- that alone already visually separates
# them from vehicles, which always carry a comet trail.
#
# Four states, not two: an earlier version only distinguished "crossing" vs
# "outside_crossing", which wrongly painted anyone off a crossing -- INCLUDING
# someone standing still on a sidewalk or curb -- as "jaywalking?". See
# classify_pedestrian_location()'s docstring: jaywalking.py's real rule
# requires being ON THE ROAD (not just off a crossing) AND sustained for
# jaywalk_min_sec, so "off_road" and "on_road" (not-yet-sustained) are now
# their own states, both distinct from the genuine "jaywalking" state.
PED_CROSSING_COLOR = (255, 255, 0)          # BGR: cyan -- inside a crossing
PED_OFF_ROAD_COLOR = (170, 170, 170)        # BGR: gray -- not on the carriageway at all, never flagged
PED_ON_ROAD_PENDING_COLOR = (60, 220, 255)  # BGR: amber -- on-road/outside-crossing, not yet sustained
PED_JAYWALKING_COLOR = (203, 0, 255)        # BGR: pink -- sustained >= jaywalk_min_sec

PED_LABELS = {
    "crossing": "crossing",
    "off_road": "off-road",
    "on_road": "on road?",
    # "?" is deliberate even now: this still isn't literally jaywalking.py's
    # event output (which is emitted only at the END of a qualifying run,
    # pooled per video) -- it's the closest a live, per-frame view can get to
    # that trigger condition. See classify_pedestrian_location()'s docstring.
    "jaywalking": "jaywalking?",
}
PED_COLORS = {
    "crossing": PED_CROSSING_COLOR,
    "off_road": PED_OFF_ROAD_COLOR,
    "on_road": PED_ON_ROAD_PENDING_COLOR,
    "jaywalking": PED_JAYWALKING_COLOR,
}


# Per-track_id persistent state for the U-turn and lane-change sub-categories --
# they need memory ACROSS calls (the track's original lane-entry heading; how many
# consecutive frames it's been reversed; which lane it sat in ~N frames ago), which
# a single rolling window can't provide. Keyed by track_id; only used when a track_id
# is passed. Cleared per video by reset_vehicle_behavior_state() (run()/run_live() call
# it at startup) so state never leaks between clips. Display-only, like the rest of
# this module -- the scored rules never touch these.
_track_original_heading: dict[int, tuple[float, float]] = {}
_track_uturn_sustain: dict[int, int] = {}
_track_lane_history: dict[int, deque] = {}
_track_last_lane: dict[int, "LaneDirection"] = {}   # last lane the track was REALLY inside (lane memory)
_track_lane_gap: dict[int, int] = {}                # consecutive detections since that real lane match (0 = in a lane now)


def reset_vehicle_behavior_state() -> None:
    """Clear all per-track classify_vehicle_behavior() state. Call before
    processing a new video (run()/run_live() do)."""
    _track_original_heading.clear()
    _track_uturn_sustain.clear()
    _track_lane_history.clear()
    _track_last_lane.clear()
    _track_lane_gap.clear()


def _prune_vehicle_behavior_state(live_track_ids) -> None:
    """Drop per-track state for tracks no longer alive (keeps memory bounded on
    long videos). `live_track_ids` is any container of currently-tracked ids."""
    live = set(live_track_ids)
    for d in (_track_original_heading, _track_uturn_sustain, _track_lane_history,
              _track_last_lane, _track_lane_gap):
        for tid in [t for t in d if t not in live]:
            d.pop(tid, None)


def classify_vehicle_behavior(
    history: deque[tuple[float, float, float]],
    scene: SceneConfig,
    turn_threshold_deg: float = TURN_THRESHOLD_DEG_DEFAULT,
    track_id: int | None = None,
) -> str:
    """Classify a vehicle's most recent behavior from its rolling
    (t_sec, cx, cy) history (oldest first), using the same lane-direction
    comparison src/rules/wrong_way.py already does (scene.point_in_any_lane
    + angle_between against lane.direction), plus a heading-change check for
    an in-progress turn. Read-only: only reads scene.lanes / scene.thresholds,
    never mutates SCENE. Deliberately its own free function (not a method,
    not baked into the render loop) so wrong_way.py / illegal_turn.py could
    import and reuse it later -- neither is modified here.

    Returns one of "aligned" | "turning" | "wrong_way" | "neutral", from TWO
    independent angle signals -- both computed from the track's stored
    (t, cx, cy) history and JITTER-REDUCED (each heading is the displacement
    between two samples VIZ_HEADING_SEG_PTS apart, never a single noisy frame
    pair):

      Signal 1  angle_dev     = angle_between(recent heading, lane.direction)
                 -- how far the car's travel deviates from the lane's legal
                 direction right now.
      Signal 2  heading_delta = angle_between(recent heading, an earlier
                 heading VIZ_ANGVEL_WINDOW_PTS points back)
                 -- the heading CHANGE RATE (angular velocity): is the car
                 actively rotating its direction of travel (mid-turn) vs
                 moving straight, independent of how the lane polygon's static
                 `direction` was defined.

    Decision (all three thresholds are named/tune-able -- see the module
    constants VIZ_ALIGNED_MAX_DEG and TURN_THRESHOLD_DEG_DEFAULT, plus the
    scored rule's own scene.thresholds["wrong_way_min_angle_deg"]):
      - angle_dev >= wrong_way_min  -> "wrong_way", UNLESS lane.allow_turn is
        True (a dedicated turn lane/pocket), where it is CAPPED to "turning"
        instead -- turn lanes never show wrong_way, but do show turning when
        the deviation is large (see LaneDirection.allow_turn's docstring).
      - else angle_dev >= VIZ_ALIGNED_MAX_DEG OR heading_delta >=
        turn_threshold_deg  -> "turning" (deviating from the lane, and/or
        actively rotating).
      - else -> "aligned" (tracking straight with the lane).

    "neutral" = not inside any mapped lane, near-stationary, or too little
    history yet -- the same cases wrong_way.py excludes before it can flag
    anything, kept as a visually distinct 4th category. Read-only: only reads
    scene.lanes / scene.thresholds, never mutates SCENE. DISPLAY-ONLY:
    src/rules/wrong_way.py's scored rule and its own thresholds are separate
    and untouched by this function.
    """
    pts = list(history)
    if len(pts) < 2:
        return "neutral"

    cx_last, cy_last = pts[-1][1], pts[-1][2]
    real_lane = scene.point_in_any_lane((cx_last, cy_last))
    if track_id is not None:
        # LANE MEMORY: a vehicle can leave its source lane and spend up to ~1.5s in
        # the open intersection CENTER (no lane polygon there, by design) mid-turn,
        # before entering a destination lane (or leaving frame). Without memory,
        # point_in_any_lane()==None there forces "neutral" and the turn label is lost.
        # So: on a REAL lane match, refresh the memory (and reset the gap counter, so
        # the remembered lane is always the MOST RECENT one, never stale from turns
        # ago); on a miss, keep using the last-known lane for up to
        # VIZ_LANE_MEMORY_FRAMES consecutive misses, then give up -> neutral.
        if real_lane is not None:
            _track_last_lane[track_id] = real_lane
            _track_lane_gap[track_id] = 0
            lane = real_lane
        else:
            gap = _track_lane_gap.get(track_id, VIZ_LANE_MEMORY_FRAMES + 1) + 1
            _track_lane_gap[track_id] = gap
            remembered = _track_last_lane.get(track_id)
            lane = remembered if (remembered is not None and gap <= VIZ_LANE_MEMORY_FRAMES) else None
    else:
        lane = real_lane
    if lane is None:
        return "neutral"

    n = len(pts)

    def heading(i_new: int, i_old: int) -> tuple[tuple[float, float], float]:
        """((dx, dy), dt) from pts[i_old] -> pts[i_new]; indices clamped in-range."""
        i_new = max(0, min(n - 1, i_new))
        i_old = max(0, min(n - 1, i_old))
        t_n, xn, yn = pts[i_new]
        t_o, xo, yo = pts[i_old]
        return (xn - xo, yn - yo), (t_n - t_o)

    seg = min(VIZ_HEADING_SEG_PTS, n - 1)   # points spanned by each heading estimate

    # recent heading (jitter-reduced over `seg` points), + speed for the stopped gate
    v_now, dt_now = heading(n - 1, n - 1 - seg)
    if dt_now <= 0:
        return "neutral"
    speed_now = (v_now[0] ** 2 + v_now[1] ** 2) ** 0.5 / dt_now
    if speed_now < scene.thresholds["stopped_speed_px_s"]:
        return "neutral"

    # Signal 1 -- deviation from the lane's legal travel direction
    angle_dev = angle_between(v_now, lane.direction)

    # Signal 2 -- heading-change rate: compare the recent heading against an
    # earlier one VIZ_ANGVEL_WINDOW_PTS points back. Needs enough history for a
    # non-overlapping earlier segment; until then it stays 0 (can't yet tell if
    # the car is rotating), so early classification leans on Signal 1 alone.
    heading_delta = 0.0
    v_ago = None   # the earlier heading vector, exposed for the turn-direction cross product below
    base = n - 1 - VIZ_ANGVEL_WINDOW_PTS
    if base - seg >= 0:
        v_ago, dt_ago = heading(base, base - seg)
        if dt_ago > 0 and (v_ago[0] or v_ago[1]):
            heading_delta = angle_between(v_now, v_ago)
        else:
            v_ago = None

    # ---- NEW (additive) per-track state: U-turn detection + lane-history tracking ----
    # Only when a track_id is supplied (run()/run_live() pass it). Captures the track's
    # ORIGINAL heading the first valid frame it's moving in a lane, counts how many
    # consecutive frames its heading has been ~reversed from that original, and records
    # the lane it's in each frame (for the lane-change check). Costs no re-derivation.
    if track_id is not None:
        original = _track_original_heading.setdefault(track_id, v_now)
        if angle_between(v_now, original) >= U_TURN_MIN_ANGLE_DEG:
            _track_uturn_sustain[track_id] = _track_uturn_sustain.get(track_id, 0) + 1
        else:
            _track_uturn_sustain[track_id] = 0
        lh = _track_lane_history.get(track_id)
        if lh is None:
            lh = _track_lane_history[track_id] = deque(maxlen=VIZ_LANE_CHANGE_WINDOW_PTS + 1)
        lh.append(lane.name)

        # U-TURN takes priority OVER wrong_way: a completed U-turn ends up opposing
        # lane.direction and would otherwise read as wrong_way. The difference is that a
        # U-turn's heading has smoothly diverged from where THIS track started (original
        # heading, captured at lane entry), sustained over many frames; a wrong_way
        # vehicle entered already opposed, so its heading never diverges from its own
        # original -> the sustain counter never accumulates. See U_TURN_* constants.
        if _track_uturn_sustain.get(track_id, 0) >= U_TURN_MIN_SUSTAIN_FRAMES:
            return "u_turn"

    # ---- EXISTING wrong_way check (unchanged) ----
    if angle_dev >= scene.thresholds["wrong_way_min_angle_deg"]:
        # opposing the lane's direction: a real wrong-way violation on a normal
        # lane, but an expected legal manoeuvre in a turn lane -> cap at "turning"
        return "turning" if lane.allow_turn else "wrong_way"

    # ---- EXISTING generic-turning condition, now SPLIT into left/right ----
    # (same trigger as before: angle_dev >= VIZ_ALIGNED_MAX_DEG OR heading_delta >=
    # turn_threshold_deg -> a turn. We just name the direction when we can.)
    if angle_dev >= VIZ_ALIGNED_MAX_DEG or heading_delta >= turn_threshold_deg:
        if v_ago is not None:
            # cross product of the UNIT earlier vs current heading == sin(turn angle);
            # sign gives rotation direction. In image coords (x right, y DOWN), a positive
            # cross is a counter-clockwise-on-screen swing, which for a forward-moving
            # vehicle reads as a LEFT turn; negative -> right. Ambiguous near 0 -> keep
            # the generic "turning" (see VIZ_TURN_CROSS_EPSILON).
            e = unit_vector(v_ago)
            r = unit_vector(v_now)
            cross = e[0] * r[1] - e[1] * r[0]
            if cross > VIZ_TURN_CROSS_EPSILON:
                return "turning_left"
            if cross < -VIZ_TURN_CROSS_EPSILON:
                return "turning_right"
        return "turning"

    # ---- NEW lane-change check (only reachable in the would-be-"aligned" case:
    #      heading stayed lane-aligned -- angle_dev low AND heading_delta low) ----
    if track_id is not None:
        lh = _track_lane_history.get(track_id)
        if lh is not None and len(lh) > VIZ_LANE_CHANGE_WINDOW_PTS:
            old_lane_name = lh[0]                      # ~VIZ_LANE_CHANGE_WINDOW_PTS detections ago
            if old_lane_name is not None and old_lane_name != lane.name:
                old_lane = next((L for L in scene.lanes if L.name == old_lane_name), None)
                if old_lane is not None and angle_between(old_lane.direction, lane.direction) < VIZ_LANE_ADJACENT_MAX_ANGLE_DEG:
                    # which side of the OLD lane's centerline is the vehicle now on?
                    # side_of_line(pt, c, c+dir): in image coords (y down), a point to
                    # screen-UP of a rightward-pointing lane gives side < 0, which reads
                    # as the vehicle's LEFT -> "lane_change_left"; side > 0 -> right.
                    xs = [p[0] for p in old_lane.polygon]; ys = [p[1] for p in old_lane.polygon]
                    c = (sum(xs) / len(xs), sum(ys) / len(ys))
                    c2 = (c[0] + old_lane.direction[0], c[1] + old_lane.direction[1])
                    side = side_of_line((cx_last, cy_last), c, c2)
                    if side < 0:
                        return "lane_change_left"
                    if side > 0:
                        return "lane_change_right"

    return "aligned"


def classify_pedestrian_location(history: deque[tuple[float, float, float]], scene: SceneConfig) -> str:
    """Classify a pedestrian using the SAME two checks src/rules/jaywalking.py's
    detect() actually applies -- not just "outside a crossing". An earlier
    version of this function checked only scene.point_in_any_crossing() and
    wrongly flagged people standing still on a sidewalk/curb as
    "jaywalking?", since a sidewalk is trivially "outside every crossing"
    too. jaywalking.py itself was already correct (confirmed by reading it,
    not touched here) -- it requires BOTH:
      - on_road = scene.point_in_any_lane(pt) is not None  -- the pedestrian
        must be on the actual carriageway, not merely off a crossing. This
        is the fix for the sidewalk/curb false positive.
      - the on_road-and-not-in-crossing condition sustained continuously for
        >= scene.thresholds["jaywalk_min_sec"] -- computed here by walking
        backward through `history` to find how long the current run has
        held, mirroring jaywalking.py's run_start/run_end tracking.
    Both are read-only reuses of SceneConfig methods / scene.thresholds --
    jaywalking.py's own code is not copied or modified.

    Returns one of:
      "crossing"    inside a mapped crossing polygon
      "off_road"    not inside any mapped lane at all (sidewalk, median,
                    waiting at a corner, etc.) -- never flagged; this is
                    exactly the case that used to be misclassified
      "on_road"     on the carriageway, outside a crossing, but hasn't been
                    continuously so for >= jaywalk_min_sec yet
      "jaywalking"  on the carriageway, outside a crossing, continuously,
                    for >= jaywalk_min_sec -- the closest this live,
                    per-frame view can get to jaywalking.py's actual
                    event-triggering condition
    """
    if not history:
        return "off_road"

    t_last, cx_last, cy_last = history[-1]
    pt = (cx_last, cy_last)

    if scene.point_in_any_crossing(pt) is not None:
        return "crossing"
    if scene.point_in_any_lane(pt) is None:
        return "off_road"

    min_dur = scene.thresholds["jaywalk_min_sec"]
    run_start_t = t_last
    for t, x, y in reversed(list(history)[:-1]):
        p = (x, y)
        if scene.point_in_any_lane(p) is not None and scene.point_in_any_crossing(p) is None:
            run_start_t = t
        else:
            break
    return "jaywalking" if (t_last - run_start_t) >= min_dur else "on_road"


def draw_trail(
    frame: np.ndarray,
    history: deque[tuple[float, float, float]],
    color: tuple[int, int, int],
    drop_oldest_fraction: float = TRAIL_DROP_OLDEST_FRACTION,
    thickness_max: int = TRAIL_THICKNESS_MAX,
    taper_length: int = TRAIL_TAPER_LENGTH,
) -> None:
    """Draw a track's trail DIRECTLY onto `frame` -- the actual displayed/
    written frame, mutated in place -- with plain cv2.line() segments at
    full opaque color. No separate alpha/color overlay canvas, no
    compositing step. This replaced an earlier alpha-blended version after
    comparing against a known-working reference (MuhammadMoinFaisal/
    YOLOv8-DeepSORT-Object-Tracking, ultralytics/yolo/v8/detect/predict.py's
    draw_boxes()): that implementation draws straight onto the frame with
    cv2.line() at full color too, with NO alpha/transparency anywhere. Its
    fade effect is entirely a THICKNESS taper: `thickness =
    int(sqrt(64/(i+i)) * 1.5)` for point index i (1=newest) -- 8px at the
    newest point, decaying to ~1px by roughly the middle of its 64-point
    buffer.

    The earlier alpha-blended version here was numerically verified correct
    (composite_alpha reproduced the exact requested color at alpha=1, not a
    washed-out blend) and was independently confirmed visible in real
    extracted frames from an actual trails_output.mp4 -- so the alpha math
    was NOT the bug. The reason it read as "invisible" during normal
    playback was that every segment was a flat 1px line: 85% of the trail
    was already alpha=1 (fully opaque) by design, so the alpha layer changed
    nothing for most of the trail's length.

    A first rewrite copied the reference's inverse-sqrt falloff shape
    directly, and turned out to have the same problem again: measured on a
    REAL 150-point track, only 2 of 128 drawn segments reached thickness>=4
    and only 16 reached thickness>=2 -- because that formula's absolute
    "thick zone" is only ~10-15 points wide regardless of total trail
    length, and TRAIL_LENGTH_DEFAULT (400) is far longer than the
    reference's fixed 64-point buffer, so the same absolute thick zone is a
    much smaller, easier-to-miss fraction of a much longer line. Root cause
    was the falloff SHAPE, not thickness_max's value.

    This version instead uses a LINEAR taper over a fixed, generous
    `taper_length` (40 points): thickness_max at the newest point, ramping
    straight down to 1px by `taper_length` points of age, flat 1px beyond
    that. Measured on the same real 150-point track: 37/128 segments now
    reach thickness>=2 and 21/128 reach thickness>=4 (vs 16 and 2 before) --
    a meaningfully larger, more visible "comet head" near the vehicle,
    independent of how long the overall persistent trail has grown.

    The oldest `drop_oldest_fraction` (default 15%) of the trail is simply
    NOT drawn at all -- a hard cutoff, replacing the old alpha fade-in,
    per the simplification request this was rewritten for.
    """
    pts = list(history)
    n = len(pts)
    if n < 2:
        return

    start = max(1, int(n * drop_oldest_fraction))
    for i in range(start, n):
        p1 = (int(pts[i - 1][1]), int(pts[i - 1][2]))
        p2 = (int(pts[i][1]), int(pts[i][2]))
        age_from_newest = (n - 1) - i  # 0 = newest segment
        if age_from_newest >= taper_length:
            thickness = 1
        else:
            frac = age_from_newest / taper_length
            thickness = max(1, round(thickness_max - frac * (thickness_max - 1)))
        cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def draw_scene_geometry(frame: np.ndarray, scene: SceneConfig, light_states: dict[str, str] | None = None) -> None:
    """Static overlay of scene.lanes / scene.crossings / scene.stop_lines /
    scene.traffic_lights, redrawn every frame since the camera is fixed and
    the geometry never moves. Read-only: only reads scene fields, no
    mutation. SCENE stores pixel coords sized for scene.width/height; this
    normalizes back out and rescales to whatever `frame` actually is (its
    own width/height), so it's correct regardless of any --scale resize
    applied upstream -- same approach src/verify_scene.py already uses.

    `light_states`, if given, is {roi_name: 'red'|'yellow'|'green'|'unknown'}
    -- the CURRENT classification for each traffic_lights ROI, computed by
    the caller via src.rules.traffic_light.classify_light_state() on the
    actual live frame (not scene.light_history, which is the offline,
    whole-video history built for the scored red_light.py rule -- see that
    module's docstring). Missing/None means "not classified this call",
    drawn as LIGHT_STATE_COLORS['unknown'] (gray)."""
    h, w = frame.shape[:2]
    scene_w = scene.width or w
    scene_h = scene.height or h
    light_states = light_states or {}

    def to_frame_px(pt: tuple[float, float]) -> tuple[int, int]:
        return (int(pt[0] / scene_w * w), int(pt[1] / scene_h * h))

    for lane in scene.lanes:
        pts = np.array([to_frame_px(p) for p in lane.polygon], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=SCENE_LANE_COLOR, thickness=2, lineType=cv2.LINE_AA)
        if len(pts):
            cv2.putText(frame, lane.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        SCENE_LANE_COLOR, 1, cv2.LINE_AA)

    for c in scene.crossings:
        pts = np.array([to_frame_px(p) for p in c.polygon], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=SCENE_CROSSING_COLOR, thickness=2, lineType=cv2.LINE_AA)

    for sl in scene.stop_lines:
        cv2.line(frame, to_frame_px(sl.p1), to_frame_px(sl.p2), SCENE_STOPLINE_COLOR, 3, lineType=cv2.LINE_AA)

    for tl in scene.traffic_lights:
        x1, y1, x2, y2 = tl.box
        (px1, py1), (px2, py2) = to_frame_px((x1, y1)), to_frame_px((x2, y2))
        cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
        radius = max(6, (abs(px2 - px1) + abs(py2 - py1)) // 4)
        state = light_states.get(tl.name, "unknown")
        color = LIGHT_STATE_COLORS.get(state, LIGHT_STATE_COLORS["unknown"])
        cv2.circle(frame, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{tl.name}: {state}", (px1, max(0, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def classify_recent(history: deque[tuple[float, float, float]], scene: SceneConfig,
                     turn_threshold_deg: float, track_id: int | None = None) -> str:
    """classify_vehicle_behavior() unchanged -- but called with only the most
    recent CLASSIFY_WINDOW points, never the full (now much longer, up to
    TRAIL_LENGTH_DEFAULT) persistent display trail. Without this, growing the
    display trail for persistence would silently change classify_vehicle_
    behavior's actual outputs (its "earliest vs current heading" comparison
    would start reading from history[0] = the track's very first point ever,
    possibly minutes old, instead of ~1s ago) even though its code is
    untouched. This keeps that comparison window exactly as it was.

    `track_id` is forwarded so classify_vehicle_behavior can keep the per-track
    U-turn / lane-change state (which needs memory beyond this rolling window)."""
    window = list(history)[-CLASSIFY_WINDOW:]
    return classify_vehicle_behavior(window, scene, turn_threshold_deg, track_id=track_id)


def prune_stale_tracks(
    histories: dict[int, deque],
    last_seen: dict[int, int],
    detect_call_idx: int,
    max_age: int = PRUNE_AFTER_DETECTIONS,
) -> None:
    """Drop a track's history only once it hasn't appeared in `max_age`
    DETECTION calls (not a short display timer) -- bounds total memory across
    a long video (many distinct track IDs over minutes) without ever cutting
    off a trail that's still actively being tracked."""
    stale = [tid for tid, seen in last_seen.items() if detect_call_idx - seen > max_age]
    for tid in stale:
        histories.pop(tid, None)
        last_seen.pop(tid, None)


def draw_box_and_label(frame: np.ndarray, p: TrackPoint, color: tuple[int, int, int], behavior: str) -> None:
    x1, y1 = int(p.cx - p.w / 2), int(p.cy - p.h / 2)
    x2, y2 = int(p.cx + p.w / 2), int(p.cy + p.h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    label = f"#{p.track_id} {p.cls} [{behavior}]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)


def draw_legend(frame: np.ndarray, show_scene: bool = False) -> None:
    # "person: ..." prefix so these never read as the same entry as the scene
    # overlay's own "crossing" (the magenta crossing-POLYGON outline, below)
    entries = list(BEHAVIOR_COLORS.items()) + [
        (REDLIGHT_LABEL, REDLIGHT_COLOR),
        ("person: crossing", PED_CROSSING_COLOR),
        ("person: off-road", PED_OFF_ROAD_COLOR),
        ("person: on road?", PED_ON_ROAD_PENDING_COLOR),
        ("person: jaywalking?", PED_JAYWALKING_COLOR),
    ]
    if show_scene:
        entries += [("lane", SCENE_LANE_COLOR), ("crossing", SCENE_CROSSING_COLOR), ("stop_line", SCENE_STOPLINE_COLOR)]
        entries += [(f"light: {s}", c) for s, c in LIGHT_STATE_COLORS.items()]
    x, y = 10, frame.shape[0] - 10 - 22 * len(entries)
    for label, color in entries:
        cv2.rectangle(frame, (x, y), (x + 18, y + 18), color, -1)
        cv2.putText(frame, label, (x + 24, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 22


def run(
    video_path: str,
    output_dir: str,
    max_frames: int | None,
    stride: int,
    model: str,
    device: str,
    trail_length: int,
    turn_threshold_deg: float,
) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SCENE.lanes:
        print("warning: SCENE.lanes is empty -- every vehicle will classify as 'neutral' "
              "(no lane to compare against). This still runs fine, just flagging it up front.")
    if not SCENE.crossings:
        print("warning: SCENE.crossings is empty -- every pedestrian will classify as "
              "'outside_crossing' (jaywalking?). This still runs fine, just flagging it up front.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open {video_path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path}: {img_w}x{img_h} @ {fps:.2f} fps, device={device}, model={model}, stride={stride}, "
          f"trail_length={trail_length}, turn_threshold_deg={turn_threshold_deg}"
          + (f", max_frames={max_frames}" if max_frames else ""))

    tracker = VideoTracker(model_path=model, device=device)
    reset_vehicle_behavior_state()  # clear per-track U-turn/lane-change state from any prior run

    # fourcc="mp4v" (the OpenCV-docs-typical choice) silently produces an
    # MPEG-4 Part 2 ("mpeg4"/"mp4v") stream via OpenCV's FFmpeg backend --
    # ffmpeg/OpenCV can read it back fine, but QuickTime Player (the default
    # macOS double-click handler for .mp4) can't decode it and shows nothing,
    # with no error. "avc1" makes OpenCV's FFmpeg backend fall back to a real
    # H.264 encoder (confirmed via ffprobe: codec_name=h264, tag=avc1), which
    # every mainstream player handles.
    out_path = out_dir / "trails_output.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                              fps / max(1, stride), (img_w, img_h))

    histories: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
    last_seen: dict[int, int] = {}                       # track_id -> detect_call_idx it last appeared in
    per_draw_counts: Counter[str] = Counter()          # every (frame, track) vehicle classification instance drawn
    per_track_last: dict[int, str] = {}                 # each VEHICLE track's most recent classification
    ped_histories: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=PED_HISTORY_LENGTH))
    ped_last_seen: dict[int, int] = {}                    # separate from `last_seen` -- see prune_stale_tracks() call below
    per_ped_counts: Counter[str] = Counter()             # every (frame, track) pedestrian classification instance drawn
    per_ped_last: dict[int, str] = {}                    # each PEDESTRIAN track's most recent classification
                                                          # (kept separate from per_track_last so the vehicle-behavior
                                                          # summary below isn't polluted by pedestrian statuses --
                                                          # track_ids are globally unique across classes -- verified
                                                          # empirically, no person id ever collided with a vehicle id --
                                                          # so this split is purely for summary-printing clarity)

    frame_idx = 0
    sampled = 0
    detect_call_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            t_sec = frame_idx / fps
            result = tracker.track_frame(frame)
            points = tracker.result_to_points(result, frame_idx, t_sec)
            detect_call_idx += 1

            for p in points:
                if p.cls in VEHICLE_CLASSES:
                    histories[p.track_id].append((p.t_sec, p.cx, p.cy))
                    last_seen[p.track_id] = detect_call_idx
                    behavior = classify_recent(histories[p.track_id], SCENE, turn_threshold_deg, track_id=p.track_id)
                    draw_trail(frame, histories[p.track_id], BEHAVIOR_COLORS[behavior])  # draws directly onto frame
                    per_draw_counts[behavior] += 1
                    per_track_last[p.track_id] = behavior
                elif p.cls in PEDESTRIAN_CLASSES:
                    # no trail drawn -- box+label only, see PED_* comment -- but we DO keep a
                    # short position history so classify_pedestrian_location() can compute
                    # jaywalking.py's sustained-duration check. Store the GROUND point
                    # (box bottom-center), not the raw box center -- see ground_point()'s
                    # docstring for why box-center falsely reads as "off the crossing" for
                    # pedestrians at this camera angle. Vehicles are untouched (still use
                    # raw p.cx/p.cy a few lines up).
                    gx, gy = ground_point(p.cx, p.cy, p.h)
                    ped_histories[p.track_id].append((p.t_sec, gx, gy))
                    ped_last_seen[p.track_id] = detect_call_idx
                    status = classify_pedestrian_location(ped_histories[p.track_id], SCENE)
                    per_ped_counts[status] += 1
                    per_ped_last[p.track_id] = status
            prune_stale_tracks(histories, last_seen, detect_call_idx)
            prune_stale_tracks(ped_histories, ped_last_seen, detect_call_idx)
            _prune_vehicle_behavior_state(histories.keys())

            for p in points:
                if p.cls in VEHICLE_CLASSES:
                    behavior = per_track_last.get(p.track_id, "neutral")
                    draw_box_and_label(frame, p, BEHAVIOR_COLORS[behavior], behavior)
                elif p.cls in PEDESTRIAN_CLASSES:
                    status = per_ped_last.get(p.track_id, "off_road")
                    if status == "off_road":
                        continue  # display filter only -- classify_pedestrian_location() already ran
                                  # above and per_ped_counts/per_ped_last already recorded this status;
                                  # this just skips drawing the box/label, nothing else
                    draw_box_and_label(frame, p, PED_COLORS[status], PED_LABELS[status])
            draw_legend(frame)

            writer.write(frame)
            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break
        frame_idx += 1
    cap.release()
    writer.release()

    size_kb = out_path.stat().st_size / 1024
    print(f"\nprocessed {sampled} sampled frame(s), {len(histories)} distinct vehicle track(s)")
    print(f"wrote {out_path} ({size_kb:.0f} KB)")

    total = sum(per_draw_counts.values())
    print("\nclassification distribution, counted per (frame, track) instance actually drawn:")
    for behavior in ("aligned", "turning", "turning_left", "turning_right", "lane_change_left", "lane_change_right", "u_turn", "wrong_way", "neutral"):
        n = per_draw_counts.get(behavior, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {behavior:<10} {n:>5}  ({pct:5.1f}%)")

    track_behaviors = Counter(per_track_last.values())
    print("\ndistinct tracks by their LAST-SEEN classification "
          f"({len(per_track_last)} tracks that ever entered a mapped lane):")
    for behavior in ("aligned", "turning", "turning_left", "turning_right", "lane_change_left", "lane_change_right", "u_turn", "wrong_way", "neutral"):
        print(f"  {behavior:<10} {track_behaviors.get(behavior, 0)}")

    ped_total = sum(per_ped_counts.values())
    print(f"\npedestrian classification distribution, counted per (frame, track) instance actually drawn "
          f"({len(per_ped_last)} distinct pedestrian track(s)):")
    for status in ("crossing", "off_road", "on_road", "jaywalking"):
        n = per_ped_counts.get(status, 0)
        pct = 100 * n / ped_total if ped_total else 0.0
        print(f"  {status:<17} {n:>5}  ({pct:5.1f}%)")
    ped_last_status = Counter(per_ped_last.values())
    print("distinct pedestrians by their LAST-SEEN classification:")
    for status in ("crossing", "off_road", "on_road", "jaywalking"):
        print(f"  {status:<17} {ped_last_status.get(status, 0)}")

    return 0


def run_live(
    video_path: str,
    stride: int,
    model: str,
    device: str,
    trail_length: int,
    turn_threshold_deg: float,
    scale: float,
    show_scene: bool,
) -> int:
    """Screen-only preview: cv2.imshow() the trails live, frame by frame,
    until the video ends or 'q' is pressed. Writes NOTHING to disk -- no
    trails_output.mp4, no heatmap, no output-dir even gets created. Purely
    for eyeballing on a machine with a real display; requires one (needs
    a working cv2 highgui backend with an attached screen).

    Performance trade-offs (both are real quality/speed trades, not free):
      --scale < 1.0   YOLO runs on a downscaled frame (roughly scale^2 fewer
                       pixels -> real CPU speedup). Cost: small/distant
                       vehicles get fewer pixels to be detected from (lower
                       recall for them), and box edges are less precise
                       (localized from fewer source pixels). The displayed
                       window is also physically smaller (this shows the
                       resized frame directly, it does not upscale back).
      --stride > 1     YOLO only runs every Nth raw frame (~Nx fewer
                       inference calls -> real CPU speedup). Cost: on the
                       frames in between, the last known box positions are
                       simply HELD STILL and redrawn on the new video frame
                       underneath them, not interpolated -- so boxes visibly
                       lag behind the actual vehicle position between
                       detections, increasingly so as --stride grows or
                       vehicles move faster. The trail's point density also
                       drops (fewer new points per second of video), making
                       the comet trail visibly coarser/chunkier.
    The video image itself is still redrawn and shown every raw frame either
    way (that part was previously also broken -- skipped frames didn't call
    cv2.imshow() at all, so the window visibly froze for stride-1 out of
    every stride frames; that's fixed here too).
    """
    if not SCENE.lanes:
        print("warning: SCENE.lanes is empty -- every vehicle will classify as 'neutral' "
              "(no lane to compare against). This still runs fine, just flagging it up front.")
    if not SCENE.crossings:
        print("warning: SCENE.crossings is empty -- every pedestrian will classify as "
              "'outside_crossing' (jaywalking?). This still runs fine, just flagging it up front.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open {video_path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w, proc_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
    print(f"{video_path}: {img_w}x{img_h} @ {fps:.2f} fps -> processing/display at {proc_w}x{proc_h} "
          f"(scale={scale}), device={device}, model={model}, stride={stride}, trail_length={trail_length}, "
          f"turn_threshold_deg={turn_threshold_deg}, show_scene={show_scene} "
          f"[LIVE PREVIEW -- nothing written to disk; press 'q' in the window to quit early]")
    if scale < 1.0 or stride > 1:
        print("  trade-off: see run_live()'s docstring -- --scale trades detection recall/box precision "
              "for speed, --stride trades box freshness/trail density for speed. Neither is free.")

    tracker = VideoTracker(model_path=model, device=device)
    reset_vehicle_behavior_state()  # clear per-track U-turn/lane-change state from any prior run
    histories: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
    last_seen: dict[int, int] = {}                       # track_id -> detect_call_idx it last appeared in
    ped_histories: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=PED_HISTORY_LENGTH))
    ped_last_seen: dict[int, int] = {}
    # LIVE red_light violation state (display-only). Mirrors src/rules/red_light.py's
    # crossing logic -- a vehicle crossing a stop_line (side_of_line sign flip) while
    # its governing traffic light reads 'red' -- but computed incrementally per frame
    # here (the scored red_light.py runs on whole trajectories in the batch pipeline;
    # --live is streaming). Does not import or call red_light.py; that rule is untouched.
    veh_prev_side: dict[int, dict[str, float]] = {}      # track_id -> {stop_line.id: last side_of_line sign source}
    veh_redlight: dict[int, float] = {}                  # track_id -> t_sec first flagged (persists for the track)

    win = "visualize_trails --live  (q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    detected_frames = 0
    displayed_frames = 0
    quit_early = False
    cached_points: list[TrackPoint] = []
    cached_behaviors: dict[int, str] = {}
    cached_ped_status: dict[int, str] = {}
    cached_light_states: dict[int, str] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame

        if frame_idx % stride == 0:
            t_sec = frame_idx / fps
            result = tracker.track_frame(proc_frame)
            points = tracker.result_to_points(result, frame_idx, t_sec)
            detected_frames += 1

            # Classify each traffic light on THIS frame first, so the live
            # red_light crossing check below reads the current light state (not
            # the previous detection's). Against the RAW full-res `frame` -- not
            # proc_frame -- since TrafficLightROI.box is stored in SCENE's full-res
            # pixel space; using proc_frame would reproduce the --scale mismatch.
            cached_light_states = {tl.name: classify_light_state(frame, tl) for tl in SCENE.traffic_lights}

            frame_behaviors: dict[int, str] = {}
            frame_ped_status: dict[int, str] = {}
            inv = 1.0 / scale  # proc-frame(scaled) -> full-res, matching SCENE geometry
            for p in points:
                if p.cls in VEHICLE_CLASSES:
                    histories[p.track_id].append((p.t_sec, p.cx, p.cy))
                    last_seen[p.track_id] = detected_frames
                    frame_behaviors[p.track_id] = classify_recent(histories[p.track_id], SCENE, turn_threshold_deg, track_id=p.track_id)
                    # LIVE red_light check: has this vehicle just crossed a stop_line
                    # (side_of_line sign flip vs last detection) while the governing
                    # light is red? Same logic as src/rules/red_light.py, done
                    # incrementally. Vehicle point scaled to full-res so it's in the
                    # SAME space as SCENE's stop_line coords (the --scale mismatch fix).
                    if p.track_id not in veh_redlight:
                        fx, fy = p.cx * inv, p.cy * inv
                        prev_sides = veh_prev_side.setdefault(p.track_id, {})
                        for sl in SCENE.stop_lines:
                            side_now = side_of_line((fx, fy), sl.p1, sl.p2)
                            prev = prev_sides.get(sl.id)
                            crossed = (prev is not None and prev != 0 and side_now != 0
                                       and (prev > 0) != (side_now > 0)
                                       # scope to the painted SEGMENT, not its infinite line --
                                       # parity with src/rules/red_light.py; without this,
                                       # right-side vehicles crossing the line's extension
                                       # (audit: #404/#515/#920/#983, all in_seg=False) falsely flag.
                                       and _crossing_within_segment((fx, fy), sl.p1, sl.p2))
                            if crossed:
                                light = SCENE.nearest_traffic_light(sl)
                                if light is not None and cached_light_states.get(light.name) == "red":
                                    veh_redlight[p.track_id] = p.t_sec
                            prev_sides[sl.id] = side_now
                elif p.cls in PEDESTRIAN_CLASSES:
                    # ground point (box bottom-center), not raw box center -- see
                    # ground_point()'s docstring / the matching comment in run() above.
                    # Detection ran on the downscaled proc_frame, so (cx,cy,h) are in
                    # proc-frame (scaled) space; scale the ground point back UP to
                    # full-res so classify_pedestrian_location compares it against
                    # SCENE's full-res polygons in the SAME coordinate space. Without
                    # this every pedestrian falsely reads "off_road" (their scaled-down
                    # coords land outside the full-res crossing/lane polygons) and the
                    # off_road display filter below then hides ALL of them. No-op when
                    # scale==1.0. Only the CLASSIFICATION point is scaled; the box is
                    # still drawn from p.cx/p.cy in proc space (see the drawing loop).
                    inv = 1.0 / scale
                    gx, gy = ground_point(p.cx, p.cy, p.h)
                    ped_histories[p.track_id].append((p.t_sec, gx * inv, gy * inv))
                    ped_last_seen[p.track_id] = detected_frames
                    frame_ped_status[p.track_id] = classify_pedestrian_location(ped_histories[p.track_id], SCENE)
            cached_points = points
            cached_behaviors = frame_behaviors
            cached_ped_status = frame_ped_status
            prune_stale_tracks(histories, last_seen, detected_frames)
            prune_stale_tracks(ped_histories, ped_last_seen, detected_frames)
            _prune_vehicle_behavior_state(histories.keys())
            # keep veh_prev_side bounded to live tracks; veh_redlight persists
            # (a violation, once seen, stays flagged for the rest of the track)
            for tid in list(veh_prev_side):
                if tid not in last_seen or detected_frames - last_seen[tid] > PRUNE_AFTER_DETECTIONS:
                    veh_prev_side.pop(tid, None)

        # drawing + display happen every raw frame, using whatever the most
        # recently DETECTED points/behaviors were (== this frame's, if we
        # just ran detection above; otherwise the stale cache from up to
        # stride-1 frames ago -- see the trade-off note in the docstring).
        # draw_trail()/draw_scene_geometry()/draw_box_and_label() all mutate
        # proc_frame directly now -- no separate overlay/alpha compositing.
        for p in cached_points:
            if p.cls not in VEHICLE_CLASSES:
                continue
            behavior = cached_behaviors.get(p.track_id, "neutral")
            draw_trail(proc_frame, histories[p.track_id], BEHAVIOR_COLORS[behavior])

        if show_scene:
            draw_scene_geometry(proc_frame, SCENE, light_states=cached_light_states)
        for p in cached_points:
            if p.cls in VEHICLE_CLASSES:
                if p.track_id in veh_redlight:
                    # red-light violation overrides the behavior label -- red box + "RED LIGHT"
                    draw_box_and_label(proc_frame, p, REDLIGHT_COLOR, REDLIGHT_LABEL)
                else:
                    behavior = cached_behaviors.get(p.track_id, "neutral")
                    draw_box_and_label(proc_frame, p, BEHAVIOR_COLORS[behavior], behavior)
            elif p.cls in PEDESTRIAN_CLASSES:
                status = cached_ped_status.get(p.track_id, "off_road")
                if status == "off_road":
                    continue  # display filter only -- see the matching comment in run() above
                draw_box_and_label(proc_frame, p, PED_COLORS[status], PED_LABELS[status])
        draw_legend(proc_frame, show_scene)

        cv2.imshow(win, proc_frame)
        displayed_frames += 1

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            quit_early = True
            break

    cap.release()
    cv2.destroyAllWindows()
    status = "quit early ('q' pressed)" if quit_early else "video ended"
    print(f"\nlive preview stopped ({status}): {displayed_frames} frame(s) displayed, "
          f"{detected_frames} of those actually ran detection (stride={stride}). Nothing was written to disk.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to a source .mp4")
    ap.add_argument("--output-dir", default=None, help="required unless --live is set")
    ap.add_argument("--live", action="store_true",
                     help="preview live in a cv2 window instead of writing output files; "
                          "ignores --max-frames and runs until the video ends or 'q' is pressed; "
                          "writes nothing to disk")
    ap.add_argument("--max-frames", type=int, default=None, help="cap sampled frames (CPU smoke test; ignored with --live)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="cpu", help="'cpu' locally, 'cuda' on Colab/Kaggle")
    ap.add_argument("--trail-length", type=int, default=TRAIL_LENGTH_DEFAULT,
                     help="persistent trail cap per track, in detected points (default 400, "
                          "~13s at stride=1/30fps); the track's full path is kept up to this many "
                          "points, not a short rolling window -- see TRAIL_LENGTH_DEFAULT's comment")
    ap.add_argument("--turn-threshold-deg", type=float, default=TURN_THRESHOLD_DEG_DEFAULT,
                     help="heading change over the trail window that counts as 'turning'")
    ap.add_argument("--scale", type=float, default=0.5,
                     help="--live only: resize frames to this fraction before detection/display, "
                          "for real-time-watchable CPU playback (default 0.5; trade-off explained "
                          "in run_live()'s docstring / printed at startup)")
    ap.add_argument("--show-scene", dest="show_scene", action="store_true", default=True,
                     help="--live only: overlay scene.lanes/crossings/stop_lines (default on)")
    ap.add_argument("--no-show-scene", dest="show_scene", action="store_false",
                     help="--live only: turn off the scene-geometry overlay")
    args = ap.parse_args()

    # run()/run_live() read the module-level SCENE; point it at this video's geometry.
    global SCENE
    SCENE = scene_for(args.video)

    if args.live:
        if args.max_frames is not None:
            print(f"note: --max-frames {args.max_frames} is ignored in --live mode "
                  f"(live preview always runs to the end of the video, or until you press 'q')")
        if args.output_dir is not None:
            print(f"note: --output-dir {args.output_dir!r} is ignored in --live mode (nothing is written to disk)")
        return run_live(args.video, args.stride, args.model, args.device,
                         args.trail_length, args.turn_threshold_deg, args.scale, args.show_scene)

    if args.output_dir is None:
        ap.error("--output-dir is required unless --live is set")
    return run(args.video, args.output_dir, args.max_frames, args.stride, args.model, args.device,
               args.trail_length, args.turn_threshold_deg)


if __name__ == "__main__":
    raise SystemExit(main())
