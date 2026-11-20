# Vehicle Monitor

A Python desktop app for monitoring vehicles in recorded traffic and parking footage. Draw a zone, count confirmed entries and exits, and review events in SQLite. Plate reading and Gmail alerts are optional.

YOLOv8s detects vehicles and ByteTrack associates them across frames. The app confirms crossings after three observations beyond the zone boundary, preserves state across compatible short tracking gaps, and uses repeated OCR readings to confirm plates.

## Run locally

Use Python 3.13 with Tkinter. Install a compatible PyTorch build first if you want NVIDIA GPU acceleration.

```sh
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS), then run:

```sh
python -m pip install -r requirements.txt
python vehicle_app.py --weights yolov8s.pt
```

Ultralytics downloads the pretrained vehicle weights if needed. Select **Open video**, draw and confirm the zone, then press **Start**. Leave the starting count blank to estimate it from the first frame, or enter a known count. Use **History** to review crossings and open saved entry snapshots.

For the preset demos, download the clips and place them at these paths:

| Preset | Local path |
| --- | --- |
| British Highway | `datasets/road-traffic-video-monitoring/british_highway_traffic.mp4` |
| Parking | `datasets/ParkingVideo.mp4` |

## Optional plate reading

Install the CPU inference dependencies:

```sh
python -m pip install onnxruntime rich pyyaml tqdm
python -m pip install --no-deps fast-alpr==0.4.0 fast-plate-ocr==1.1.0 open-image-models==0.6.0
```

The second command keeps the existing OpenCV build, which supports the CLI demo windows. Put these files in `plate_models/`:

- [Plate detector](https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx)
- [CCT OCR weights](https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx)
- [CCT OCR configuration](https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml)

Enable **Read plates**. The Parking preset enables it by default. The app filters poor crops and confirms a plate after at least three agreeing reads with at least 75% agreement among recent accepted reads.

Email is off by default. To enable it, use a [Gmail App Password](https://support.google.com/accounts/answer/185833) in the app's form. Inbox delivery was not verified in the project evaluation.

## Evaluation and sources

The first 790 frames of British Highway produced 10 entries and 10 exits. The parking clip showed the same confirmed plate on exit and return. These are functional checks on selected clips. The zone assumes a fixed camera, and recovery is limited to short gaps.

- [ByteTrack paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)
- [British Highway video collection](https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring)
- [Intel ParkingVideo sample](https://huggingface.co/Intel/license-plate-recognition/tree/main)
