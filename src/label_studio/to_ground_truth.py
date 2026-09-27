"""src/label_studio/to_ground_truth.py — Label Studio JSON export -> my_labels.json.

Converts timeline-label annotations (config: src/label_studio/label_config.xml)
into the exact ground_truth.json shape evaluate.py reads:
    {"C3896.MP4": {"duration": 340.34, "fps": 29.97, "events": [[s, e, label], ...]}}

Keys are the ORIGINAL file names from samples/proxies/index.json (what
run_submission.py writes into predictions.json), not the proxy names Label
Studio saw. Frame numbers are 1-based in Label Studio; a range of frames
[a, b] becomes [(a-1)/fps, b/fps] seconds, i.e. the time those frames cover.

A task that was submitted with no regions counts as "labelled, no events" and
is kept with an empty list; a task nobody submitted is skipped with a warning.

    python -m src.label_studio.to_ground_truth --export export.json --index samples/proxies/index.json --out my_labels.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import OFFICIAL_CLASSES  # noqa: E402

UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")  # Label Studio prefixes uploaded files with a short hash


def video_stem(task: dict) -> str:
    """Uploaded file ("a1b2c3d4-C3896.mp4") or local-storage URL
    ("/data/local-files/?d=Westminster-%5Csamples%5Cproxies%5CC3896.mp4") -> "C3896"."""
    name = task.get("file_upload") or unquote(str(task["data"].get("video", ""))).replace("\\", "/").rsplit("/", 1)[-1]
    return Path(UPLOAD_PREFIX.sub("", name)).stem.upper()


def pick_annotation(task: dict) -> dict | None:
    done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if len(done) > 1:
        print(f"  warning: {len(done)} annotations on task {task.get('id')}, using the latest", file=sys.stderr)
    return max(done, key=lambda a: a.get("updated_at") or a.get("created_at") or "") if done else None


def merge_same_class(events: list[list]) -> list[list]:
    """Per the task FAQ, simultaneous same-class events are one segment covering both."""
    out: list[list] = []
    for s, e, label in sorted(events, key=lambda x: (x[2], x[0])):
        if out and out[-1][2] == label and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, label])
    return sorted(out, key=lambda x: (x[0], x[2]))


def convert(tasks: list[dict], index: dict) -> dict:
    by_stem = {stem.upper(): meta for stem, meta in index.items()}
    gt: dict = {}
    for task in tasks:
        stem = video_stem(task)
        meta = by_stem.get(stem)
        if meta is None:
            print(f"  warning: task {task.get('id')} video {stem!r} not in index.json, skipped", file=sys.stderr)
            continue
        ann = pick_annotation(task)
        if ann is None:
            print(f"  warning: {meta['source']} has no submitted annotation, skipped", file=sys.stderr)
            continue

        fps, duration = meta["fps"], meta["duration"]
        events = []
        for region in ann.get("result", []):
            if region.get("type") != "timelinelabels":
                continue
            value = region["value"]
            for label in value.get("timelinelabels", []):
                if label not in OFFICIAL_CLASSES:
                    print(f"  warning: {meta['source']}: unknown label {label!r}, skipped", file=sys.stderr)
                    continue
                for r in value.get("ranges", []):
                    start = round(max(0.0, (r["start"] - 1) / fps), 3)
                    end = round(min(duration, r["end"] / fps), 3)
                    if end > start:
                        events.append([start, end, label])
        gt[meta["source"]] = {"duration": duration, "fps": fps, "events": merge_same_class(events)}
    return gt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="Label Studio export (JSON format)")
    ap.add_argument("--index", default="samples/proxies/index.json")
    ap.add_argument("--out", default="my_labels.json")
    args = ap.parse_args()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    index = json.loads(Path(args.index).read_text())
    gt = convert(tasks, index)
    Path(args.out).write_text(json.dumps(gt, indent=1))
    for name, entry in sorted(gt.items()):
        counts: dict[str, int] = {}
        for _, _, label in entry["events"]:
            counts[label] = counts.get(label, 0) + 1
        print(f"{name}: {len(entry['events'])} events {counts}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
