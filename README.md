# WIUT Hackathon 2026 — Computer Vision track: team BlockVision

Traffic events from a fixed road camera: **detect** them as time segments
(`[start_sec, end_sec, label]`, Part A) and **anticipate** accidents with a
causal per-frame risk score (Part B). Offline, open weights only,
deterministic, sized for a T4-class GPU.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+ (developed on 3.12)
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

**Weights** ship in the repo: `weights/yolov8n.pt` (6.5 MB, well under the
5 GB limit), loaded by absolute path from `solution.py`, so the run needs no
internet. `weights/download.sh` only re-fetches that same file if it is ever
missing. **ffmpeg** comes from the `imageio-ffmpeg` wheel (a bundled static
binary), so no system install is needed either.

Reproduce our sample-video output and score it against our own labels:
```bash
python run_submission.py --videos samples --out predictions_samples.json --team BlockVision
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

## Approach

```
.mp4 ──► src/video_io.py      one ffmpeg pass: every 3rd frame, downscaled to 1280 px,
          │                   + full-res crop of the traffic-light head
          ├──► src/detect.py  YOLOv8n (COCO: person, bicycle, car, motorcycle, bus, truck)
          │                   + ByteTrack → per-object trajectories in 4K pixel space
          ├──► src/rules/traffic_light.py   HSV colour classification of the signal crop
          ▼
     src/rules/*.py           one rule per class over trajectories + scene geometry
          ▼                   (lanes + legal direction, crosswalks, stop line: src/config.py)
     src/postprocess.py       merge fragments < 1 s apart, drop blips < 0.5 s,
                              no same-class overlaps, clip to video length
```

**What is learned vs. rule-based.**
- *Learned:* only the object detector — Ultralytics YOLOv8n with its public
  COCO-pretrained weights, **not fine-tuned**. We did not train anything.
- *Rule-based:* tracking (ByteTrack, a matching algorithm), traffic-light
  colour (HSV thresholds), every event class (geometric predicates over
  trajectories and the hand-drawn scene layout), and Part B.
- *Tuned by hand:* the thresholds in `src/config.py` (`SceneConfig.thresholds`),
  checked against our own labels of the sample videos (`my_labels.json`).

**Part B** (`src/risk_estimator.py`) is strictly causal: it runs its own
detector + tracker on the frames `step()` receives (every 3rd frame) and never
reads the file or Part A's output. Risk = pairwise time-to-collision between
tracked objects (≤ 5 s → high) blended with sudden-deceleration spikes,
smoothed and decaying toward 0 when nothing is happening.

**Why this design.** The camera never moves, so lanes, crosswalks and the stop
line are constants that can be written down once; most classes are then
"where did this object go, and when" questions over trajectories. With no
labels provided and ~60 self-labelled events, a rule-based pipeline is the one
we could actually validate.

**Speed.** The footage is 4K H.264 High 4:2:2 10-bit. `cv2.VideoCapture.read()`
decodes it at ~25 fps (slower than real time) and NVDEC cannot decode 4:2:2
H.264 at all, so Part A decodes once with multi-threaded ffmpeg, dropping
skipped frames before colour conversion (~95 fps on our laptop).

## Results on the sample videos

_Filled in from `evaluate.py --per-video` against `my_labels.json`._

RESULTS_PLACEHOLDER

## Determinism

Seeds are fixed in `src/detect.py:set_determinism` (`random`, `numpy`,
`torch`, CUDA). YOLO inference, ByteTrack and all rules are deterministic for a
given input, and ffmpeg decoding/scaling is bit-exact, so two runs on the same
machine give the same `predictions.json`. cuDNN kernel selection can differ
between GPU models, which may shift detection confidences in the last decimal
places on a different machine.

## Datasets, models and licences

| What | Licence | Use |
|---|---|---|
| Ultralytics YOLOv8n, COCO-pretrained weights | AGPL-3.0 | object detector (not fine-tuned) |
| ByteTrack (bundled with ultralytics) | MIT | multi-object tracking |
| FFmpeg via `imageio-ffmpeg` | LGPL/GPL (binary), BSD-2 (wrapper) | video decoding |
| Our own annotations of the 4 sample videos (`my_labels.json`) | ours | dev-set scoring only |

No external training datasets were used.

## Labeling your dev set

We label with [Label Studio](https://labelstud.io) (Apache-2.0), in its own
venv so its dependencies never touch the submission's pins:

```bash
python -m venv ../.venv-labelstudio
../.venv-labelstudio/Scripts/pip install label-studio imageio-ffmpeg opencv-python-headless

# 1. 960px H.264 copies of the 4K samples (same fps + frame count) + index.json
../.venv-labelstudio/Scripts/python -m src.label_studio.make_proxies --videos samples --out samples/proxies

# 2. start Label Studio with local-file serving (browser upload is flaky for video),
#    create a project, paste src/label_studio/label_config.xml into
#    Settings -> Labeling Interface -> Code, then Settings -> Cloud Storage ->
#    Add Source Storage -> Local files: absolute path to samples/proxies,
#    file filter .*\.mp4, Import method = "Files" (NOT "Tasks") -> Sync.
#    Label every video (submit even if it has no events).
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="<absolute path of the folder containing this repo>" \
../.venv-labelstudio/Scripts/label-studio start

# 3. Export -> JSON, then convert to evaluate.py's ground-truth shape
python -m src.label_studio.to_ground_truth --export export.json --out my_labels.json
```


## Repo layout

```
solution.py            the interface the harness calls (CLASSES, detect_events, RiskEstimator)
run_submission.py      organizers' harness (unchanged)
evaluate.py            format check + official metric (unchanged)
weights/yolov8n.pt     detector weights (shipped); download.sh re-fetches it
src/
  video_io.py          single fast ffmpeg decode pass (sampled, downscaled, + signal crop)
  detect.py            YOLO + ByteTrack -> trajectories
  config.py            scene geometry (per camera framing) + thresholds
  rules/               one rule module per class + traffic-light colour
  postprocess.py       segment cleanup for tIoU matching
  risk_estimator.py    Part B, causal
  label_studio/        dev-set labelling: proxy videos, config, export -> my_labels.json
  pick_points.py, verify_scene.py, visualize_*.py   scene-drawing and visualisation tools
my_labels.json         our labels of the sample videos (ground_truth.json format)
predictions_samples.json   our output on the sample videos
```

## Team

| Member | Role | Contributions |
|---|---|---|
| Muhammadsayyid Tursunov (muhammadsayyid.tursunov@polito.uz) | ML / pipeline | dev-set labelling (Label Studio), fast video decoding, evaluation, submission |
| Abdulaziz Farkhodov (abdulaziz.farxodov@polito.uz, [GitHub](https://github.com/Farkhodov721)) | ML / pipeline, website | detector + tracker pipeline, event rules, scene geometry, team website |
| Asadullo Ismoilov (asadullo.ismoilov@polito.uz) | Presentation | pitch and demo presentation, website support |
