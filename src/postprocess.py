"""src/postprocess.py — boundary cleanup for Part A event segments.

The metric scores tIoU up to 0.7, so sloppy start/end boundaries (a single
missed frame fragmenting one real event into three, or a blip lasting a few
frames) directly cost score. This module merges/cleans per class before
solution.detect_events() returns.
"""
from __future__ import annotations

Event = list  # [start_sec, end_sec, label]


def drop_short_segments(events: list[Event], min_dur: float = 0.5) -> list[Event]:
    return [e for e in events if e[1] - e[0] >= min_dur]


def merge_close_segments(events: list[Event], max_gap: float = 1.0) -> list[Event]:
    """Merge same-class segments whose gap is < max_gap. Assumes no overlaps
    within a class (rule modules are expected to emit non-overlapping runs)."""
    by_label: dict[str, list[Event]] = {}
    for s, e, label in events:
        by_label.setdefault(label, []).append([s, e, label])

    merged: list[Event] = []
    for label, segs in by_label.items():
        segs.sort(key=lambda x: x[0])
        cur = None
        for s, e, _ in segs:
            if cur is None:
                cur = [s, e, label]
            elif s - cur[1] < max_gap:
                cur[1] = max(cur[1], e)
            else:
                merged.append(cur)
                cur = [s, e, label]
        if cur is not None:
            merged.append(cur)
    return merged


def clip_to_duration(events: list[Event], duration: float) -> list[Event]:
    out = []
    for s, e, label in events:
        s = max(0.0, s)
        e = min(duration, e)
        if s < e:
            out.append([s, e, label])
    return out


def dedupe_overlaps_same_class(events: list[Event]) -> list[Event]:
    """Safety net: if a rule module ever emits overlapping same-class segments
    (run_submission.py would otherwise silently drop the later one), merge
    them here instead of losing recall."""
    by_label: dict[str, list[Event]] = {}
    for s, e, label in events:
        by_label.setdefault(label, []).append([s, e, label])
    out: list[Event] = []
    for label, segs in by_label.items():
        segs.sort(key=lambda x: x[0])
        cur = None
        for s, e, _ in segs:
            if cur is None:
                cur = [s, e, label]
            elif s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                out.append(cur)
                cur = [s, e, label]
        if cur is not None:
            out.append(cur)
    return out


# Per-class (merge_gap_sec, min_dur_sec), overriding the defaults below. Rules emit
# one segment per object, but the annotation convention is one segment per event
# ("two of the same class at once -> one segment covering both"), and real events
# of different classes have very different lengths. Tuned on our dev labels
# (my_labels.json), see README "Results".
CLASS_PARAMS: dict[str, tuple[float, float]] = {
    "jaywalking": (1.0, 4.0),
    "congestion": (10.0, 20.0),
    "failure_to_yield": (2.0, 0.5),
    "red_light": (3.0, 0.5),
    "stopped_vehicle": (1.0, 12.0),
}


def postprocess_events(
    events: list[Event],
    duration: float,
    min_dur: float = 0.5,
    merge_gap: float = 1.0,
) -> list[Event]:
    """Full boundary-cleanup pipeline: dedupe overlaps -> merge close fragments
    -> drop sub-threshold blips -> clip to [0, duration] -> sort. Merge gap and
    minimum duration are per class (CLASS_PARAMS), falling back to the args."""
    events = dedupe_overlaps_same_class(events)
    out: list[Event] = []
    for label in {e[2] for e in events}:
        gap, dur = CLASS_PARAMS.get(label, (merge_gap, min_dur))
        segs = [e for e in events if e[2] == label]
        out += drop_short_segments(merge_close_segments(segs, max_gap=gap), min_dur=dur)
    out = clip_to_duration(out, duration)
    out.sort(key=lambda x: (x[0], x[2]))
    return out
