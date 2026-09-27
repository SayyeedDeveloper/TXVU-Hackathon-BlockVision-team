#!/usr/bin/env bash
# weights/download.sh — fetch/cache model weights ONCE, with internet access,
# before the offline evaluation run. The judges' run itself has no internet,
# so every weight file this repo depends on must exist under weights/ (or be
# fetched here) before `python run_submission.py ...` is invoked.
#
# ultralytics auto-downloads yolov8*.pt to its cache dir on first use if it's
# missing; this script pins that download to happen here, once, and copies the
# result into weights/ so `git`/the submission archive ships a stable file
# (judges' box: no internet -> ultralytics' own lazy-download would fail).
set -euo pipefail

cd "$(dirname "$0")"

YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"

echo "== weights/download.sh: fetching ${YOLO_MODEL} =="
python3 - "$YOLO_MODEL" <<'PY'
import sys
from ultralytics import YOLO

model_name = sys.argv[1]
YOLO(model_name)  # triggers ultralytics' own download into its cache dir
PY

# Locate ultralytics' cache copy and mirror it into weights/ so it's shipped
# with the repo (weights/ is NOT gitignored; keep this under the 5 GB budget).
CACHE_PATH="$(python3 -c "
from ultralytics.utils import SETTINGS
from pathlib import Path
import sys
p = Path(SETTINGS.get('weights_dir', '.')) / sys.argv[1]
print(p if p.exists() else '')
" "$YOLO_MODEL")"

if [ -n "$CACHE_PATH" ] && [ -f "$CACHE_PATH" ]; then
    cp -f "$CACHE_PATH" "./${YOLO_MODEL}"
    echo "copied $CACHE_PATH -> weights/${YOLO_MODEL}"
elif [ -f "./${YOLO_MODEL}" ]; then
    echo "weights/${YOLO_MODEL} already present"
else
    echo "WARNING: could not locate downloaded weights for ${YOLO_MODEL}; check ultralytics cache dir manually" >&2
fi

echo "== done. Point DETECT_MODEL in solution.py at weights/${YOLO_MODEL} for a fully offline run. =="
