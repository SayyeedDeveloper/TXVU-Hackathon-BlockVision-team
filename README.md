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

Scored with the official `evaluate.py` against **our own labels** of the four
sample videos (`my_labels.json`, 59 events, labelled in Label Studio with the
task's start/end conventions). Thresholds were tuned on C3896, C3897 and C3905;
**C3902 was held out** (it also has a different camera framing).

| Class | F1@0.3 | F1@0.5 | F1@0.7 | mean | TP/FP/FN @0.5 |
|---|---|---|---|---|---|
| congestion | 0.800 | 0.800 | 0.400 | **0.667** | 2/1/0 |
| jaywalking | 0.525 | 0.230 | 0.066 | **0.273** | 7/22/25 |
| red_light | 0.182 | 0.182 | 0.000 | 0.121 | 1/9/0 |
| stopped_vehicle | 0.182 | 0.182 | 0.000 | 0.121 | 1/8/1 |
| failure_to_yield | 0.114 | 0.000 | 0.000 | 0.038 | 0/28/7 |
| illegal_turn, illegal_u_turn, near_miss, solid_line_crossing, stop_line | 0 | 0 | 0 | 0 | missed (in labels, not detected) |

**Score A = 0.122** on the dev labels (0.049 before tuning). Held-out
jaywalking F1 rose from 0.262 to 0.319 with the tuned settings, so the tuning
generalises rather than memorising the three tuning videos. Part B is not scored
on the samples: none of them contains an accident.

**What was tuned** (`src/config.py` thresholds, `src/postprocess.py` `CLASS_PARAMS`):
congestion now needs ≥ 12 slow vehicles (4 fired on every red-light queue);
jaywalking ignores < 3 s on the road and < 4 s segments; per-class merge gaps
join the per-object fragments into one event, matching the annotation convention.
Classes whose rules produced only false alarms (near_miss, wrong_way,
illegal_u_turn) are removed from `CLASSES`, which the task allows.

**Runtime** on our laptop (RTX 5060, `run_submission.py`, budget = 3× duration):

| Video | Duration | Part A | Part B | Total |
|---|---|---|---|---|
| C3896 | 340 s | 234 s | 485 s | 2.1× |
| C3897 | 318 s | 194 s | 452 s | 2.0× |
| C3902 | 318 s | 230 s | 491 s | 2.3× |
| C3905 | 128 s | 70 s | 111 s | 1.4× |

Part A times are from before the tuning, which also dropped the slow pairwise
near_miss rule (C3905 Part A: 70 s → 44 s). Most of Part B is the harness's own
`cv2` decode of every 4K 10-bit frame (~1.2× real time), which we cannot change.

**Honest failure cases.** failure_to_yield fires on many vehicle–pedestrian
overlaps that are not violations (28 false alarms); stop_line, illegal_turn and
solid_line_crossing are never detected (the scene has no solid-line or
turn-restriction geometry yet); scene geometry is hand-drawn per camera framing
and matched by file name, so a test video with an unseen framing falls back to
C3896's layout.

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
