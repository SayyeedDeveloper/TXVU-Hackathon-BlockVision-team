"""src/rules/zone_matching.py — entry/exit zone matching, a PARALLEL,
EXPERIMENTAL alternative to src/rules/wrong_way.py's frame-by-frame
heading-vs-lane-direction comparison.

Not wired into src/rules/__init__.py's RULES registry and not called by
solution.py -- this module is purely for side-by-side comparison while the
approach is being evaluated. src/rules/wrong_way.py, src/rules/jaywalking.py,
and the existing LaneDirection/lanes data are untouched by this file.

CONCEPT: instead of asking "does this vehicle's instantaneous heading oppose
its lane's direction vector, right now, in this frame" (wrong_way.py), this
looks at a vehicle's FULL trajectory once, and asks two much simpler
questions: which src.config.EntryExitZone did it first pass through (its
entry), and which did it last pass through (its exit)? The entry->exit pair
alone determines a whole-trip movement classification:
  - same zone's compass on both ends                  -> "wrong_way" (U-turn
    or driving back out the way it came in -- see EntryExitZone's docstring)
  - opposite compass edges (north<->south, east<->west) -> "straight"
  - adjacent compass edges                              -> "left_turn" or
    "right_turn", depending on rotational direction (see _turn_type below)
  - no clear entry and/or exit zone hit anywhere in the trajectory -> "unknown"

CURRENT SCENE.entry_exit_zones ARE PLACEHOLDERS (rough border-strip guesses,
not hand-picked against a real frame -- see the "*** PLACEHOLDER GEOMETRY
***" comment in src/config.py). Treat any output against them as a logic
smoke test, not a real classification, until real zones are picked with
src/pick_points.py.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track

LABEL = "zone_matching"

# clockwise compass order; used by _turn_type's rotational-distance rule
_CLOCKWISE = ["north", "east", "south", "west"]


def _turn_type(entry_compass: str, exit_compass: str) -> str:
    """Classify a movement purely from its entry/exit compass edges, using
    each edge's position in clockwise order. A vehicle entering from compass
    edge X is, geometrically, travelling in the OPPOSITE direction of X (e.g.
    entering from the west edge means heading east) -- turning left from
    that heading always lands on the next edge clockwise from the entry
    edge, turning right always lands on the next edge counter-clockwise.
    This is a fixed geometric fact for a 4-way intersection, independent of
    which side of the road traffic drives on.

    Verified case: entry="west" (heading east), exit="north" -- facing east,
    left is north -- diff = (north_idx - west_idx) % 4 = (0 - 3) % 4 = 1 ->
    "left_turn". entry="west", exit="south" -- facing east, right is south
    -- diff = (2 - 3) % 4 = 3 -> "right_turn". Both match hand-worked
    compass reasoning.
    """
    if entry_compass not in _CLOCKWISE or exit_compass not in _CLOCKWISE:
        return "unknown"
    i = _CLOCKWISE.index(entry_compass)
    j = _CLOCKWISE.index(exit_compass)
    diff = (j - i) % 4
    if diff == 0:
        return "wrong_way"
    if diff == 2:
        return "straight"
    if diff == 1:
        return "left_turn"
    return "right_turn"  # diff == 3


def classify_movement_detailed(traj, scene) -> tuple[str, str | None, str | None]:
    """Classify one vehicle's WHOLE trajectory (list[TrackPoint], already
    time-ordered -- as produced by src.detect.group_by_track) by its first
    entry-zone hit and last exit-zone hit. Returns
    (movement_label, entry_zone_name_or_None, exit_zone_name_or_None) --
    the zone names are included (beyond just the label) so callers/reports
    can show which specific placeholder/real zones drove the classification.
    """
    if not scene.entry_exit_zones or not traj:
        return "unknown", None, None

    entry_zone = None
    for p in traj:  # chronological: first ENTRY-type zone hit
        z = scene.zone_at((p.cx, p.cy))
        if z is not None and z.zone_type == "entry":
            entry_zone = z
            break

    exit_zone = None
    for p in reversed(traj):  # reverse-chronological: last EXIT-type zone hit
        z = scene.zone_at((p.cx, p.cy))
        if z is not None and z.zone_type == "exit":
            exit_zone = z
            break

    if entry_zone is None or exit_zone is None:
        return "unknown", entry_zone.name if entry_zone else None, exit_zone.name if exit_zone else None

    return _turn_type(entry_zone.compass, exit_zone.compass), entry_zone.name, exit_zone.name


def classify_movement(traj, scene) -> str:
    """Same as classify_movement_detailed but returns just the label --
    the shape item 2 of the request asked for."""
    label, _, _ = classify_movement_detailed(traj, scene)
    return label


def classify_all(trajectories, scene) -> dict[int, tuple[str, str | None, str | None]]:
    """Convenience for comparison reports: classify_movement_detailed() for
    every VEHICLE track in `trajectories` (raw list[TrackPoint] or an
    already-grouped dict[track_id, list[TrackPoint]]). Returns
    {track_id: (label, entry_zone_name, exit_zone_name)}."""
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    results = {}
    for tid, traj in by_track.items():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        results[tid] = classify_movement_detailed(traj, scene)
    return results
