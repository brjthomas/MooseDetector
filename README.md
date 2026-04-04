# MooseDetector

An edge AI system for real-time wildlife detection using a thermal camera on a Raspberry Pi 5. It runs live YOLO inference on thermal frames, tracks detected objects across frames, triggers GPIO hardware alerts, and records video with a pre-buffer for evidence capture.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Architecture](#software-architecture)
- [Project Structure](#project-structure)
- [Component Reference](#component-reference)
- [Models](#models)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Deployment (Run on Boot)](#deployment-run-on-boot)
- [GPIO Pinout](#gpio-pinout)
- [Outputs](#outputs)
- [Development Setup](#development-setup)

---

## Overview

MooseDetector captures thermal video from a Seek Thermal camera, runs YOLO object detection with BoT-SORT tracking on each frame, and:

- Drives a **GPIO alert output** (pin 17) HIGH when an animal or person is detected
- Keeps the alert active for ~3 seconds after the tracked object leaves the frame (persistence)
- Maintains a **10-second rolling pre-buffer** of raw and annotated frames
- Writes the pre-buffer plus live footage to timestamped AVI files when a **record button** (pin 23) is pressed
- Logs per-frame **performance metrics** (FPS, inference time, object count, alert state) to CSV

The system is designed to run headlessly on boot but can also display a live cv2 window when a display is connected.

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | aarch64, 64-bit OS required |
| Seek Thermal Camera | USB-connected, supported by Seek SDK v4.4.2.20 |
| GPIO alert device | Connected to GPIO pin 17 (e.g. buzzer, relay, LED) |
| Record button | Momentary switch on GPIO pin 23 |
| Recording LED | LED (+ resistor) on GPIO pin 24 |
| Hailo-8L AI HAT | Optional — see `feature-AI-HAT` branch |

---

## Software Architecture

The application runs three concurrent threads:

```
┌─────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 5                       │
│                                                             │
│  Seek Thermal Camera (USB)                                  │
│         │                                                   │
│         ▼  Seek SDK callback (SDK thread)                   │
│  ┌─────────────────┐                                        │
│  │  ThermalCamera  │  ARGB8888 frames → FrameBuffer         │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼  (blocking get, processing thread)              │
│  ┌──────────────────────────────────────────────┐          │
│  │  Processing Thread                           │          │
│  │                                              │          │
│  │  1. ARGB → RGB colour conversion             │          │
│  │  2. YOLO26 inference  (BoT-SORT tracking)    │          │
│  │  3. AlertManager  → GPIO pin 17              │          │
│  │  4. MetricsLogger → CSV + terminal           │          │
│  │  5. VideoRecorder → pre-buffer / AVI files   │          │
│  │  6. cv2 imshow (if display connected)        │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  Main Thread: prints stats every 5 seconds, handles Ctrl+C │
└─────────────────────────────────────────────────────────────┘
```

**Why two threads for capture + processing?**
The Seek SDK delivers frames via a C callback on its own thread. The processing thread blocks on a `threading.Event` and picks up the latest frame from a single-slot `FrameBuffer`. If inference is slower than the camera frame rate (9 FPS), older frames are silently overwritten — the drop rate is logged so you can monitor this.

---

## Project Structure

```
MooseDetector/
├── src/
│   ├── main.py                      # Entry point
│   └── moosedetector/
│       ├── __init__.py
│       ├── app.py                   # Application orchestrator
│       ├── config.py                # All configuration (dataclasses)
│       ├── pipeline.py              # FrameBuffer + YOLO inference + visualisation
│       ├── thermalcamera.py         # Seek SDK hardware abstraction
│       ├── alert_manager.py         # GPIO alert logic with object persistence
│       ├── metrics.py               # FPS / inference timing / CSV logging
│       ├── video_recorder.py        # Pre-buffer + button-triggered AVI recording
│       └── square_wave.py           # Standalone GPIO test utility
├── scripts/
│   ├── run_camera.py                # Placeholder
│   └── run_on_boot.sh               # Venv activation + launch script
├── deploy/
│   └── moosedetector.desktop        # GNOME autostart entry
├── models/
│   ├── yolo26_best_v1.pt            # PyTorch model (used at runtime)
│   └── yolo26_ncnn_model_v1/        # NCNN-optimised model (future use)
│       ├── metadata.yaml
│       ├── model.ncnn.bin
│       ├── model.ncnn.param
│       └── model_ncnn.py
├── docs/
│   ├── Seek_Thermal_SDK_*.pdf       # Official Seek SDK documentation
│   └── ultralytics_tracking_example.py
├── logs/                            # Auto-created; timestamped CSV metrics
├── videos/                          # Auto-created; raw + overlay AVI recordings
├── requirements.txt
└── README.md
```

---

## Component Reference

### `app.py` — Application Orchestrator

Instantiates all subsystems, starts the camera, launches the processing thread, and blocks on a 5-second stats loop. Catches `KeyboardInterrupt` for graceful cleanup.

Stats printed every 5 seconds:
```
[Stats] Frames received: 45 | Dropped: 2 | Drop rate: 4.4% | Recording: False
```

---

### `config.py` — Configuration

All tuneable parameters live here as frozen-style dataclasses with factory defaults:

| Class | Key fields |
|-------|-----------|
| `DetectionConfig` | `model_path`, `confidence` (0.5), `tracker` (BoT-SORT) |
| `AlertConfig` | `gpio_pin` (17), `alert_classes` (Animal, person), `persistence_frames` (30) |
| `MetricsConfig` | `log_path`, `terminal_logging`, `overlay_display`, `fps_smoothing` (20) |
| `DataCollectionConfig` | `button_pin` (23), `led_pin` (24), `pre_buffer_seconds` (10), `fps` (9) |

Default paths are under `/home/moose/projects/MooseDetector/`. Edit `config.py` to change them.

---

### `pipeline.py` — Frame Pipeline

- **`FrameBuffer`** — thread-safe single-slot buffer; newer frames overwrite unprocessed ones. Signals the processing thread via `threading.Event`.
- **Colour conversion** — Seek SDK delivers `ARGB8888`; this is re-ordered to `RGB` before YOLO sees it.
- **YOLO tracking** — calls `model.track(frame, persist=True, tracker="botsort.yaml")`. Track IDs are stable within a continuous detection sequence.
- **Detection extraction** — returns a list of dicts: `{track_id, class_name, confidence, bbox}`.
- **Visualisation** — draws bounding boxes, track IDs, class labels, and an optional metrics overlay using `cv2`.

---

### `thermalcamera.py` — Hardware Abstraction

The only file that directly calls the Seek SDK. Handles:
- Camera `CONNECT` / `DISCONNECT` / `ERROR` events
- Frame conversion to `COLOR_ARGB8888` format
- Forwarding each frame to `FrameBuffer` via the SDK callback

Isolating the SDK here means the rest of the codebase can be tested on any machine that can produce BGR/RGB image arrays.

---

### `alert_manager.py` — GPIO Alert

Drives GPIO pin 17 to signal external hardware when a target class is detected.

- **Persistence**: each track ID gets a counter reset to `persistence_frames` (30) on detection. The counter decrements each frame. The pin goes LOW only when all active counters reach zero. At 9 FPS this is ~3.3 seconds.
- **Class filter**: only classes listed in `AlertConfig.alert_classes` (default: `Animal`, `person`) trigger the alert. `car` is detected but does not trigger.
- Registers an `atexit` handler to call `GPIO.cleanup()`.

---

### `metrics.py` — Performance Metrics

| Class | Purpose |
|-------|---------|
| `FrameMetrics` | Dataclass: timestamp, FPS, inference time breakdown, object count, alert state |
| `PerformanceMetrics` | Calculates smoothed FPS from a 20-frame interval deque; extracts YOLO preprocess / inference / postprocess timings |
| `MetricsLogger` | Writes a new `metrics_YYYYMMDD_HHMMSS.csv` each run; prints a formatted line to the terminal every 30 frames |
| `MetricsOverlay` | Draws FPS, object count, and a red `ALERT` indicator onto the frame |

Terminal output format:
```
[Frame 120] FPS: 8.7 | Inference: 22.3ms | Objects: 1 | Alert: YES
```

---

### `video_recorder.py` — Pre-Buffer Recording

- Maintains a `collections.deque` circular buffer holding the last ~90 frames (10 s × 9 FPS) as both raw RGB and overlay frames.
- A `gpiozero.Button` on pin 23 toggles recording. When recording starts, the pre-buffer is flushed first so the AVI file includes footage before the button was pressed.
- A `gpiozero.LED` on pin 24 blinks while recording.
- Two `cv2.VideoWriter` files are created per recording session:
  - `raw_YYYYMMDD_HHMMSS.avi` — thermal image without annotations
  - `overlay_YYYYMMDD_HHMMSS.avi` — same image with YOLO bounding boxes and metrics
- The FPS written into the AVI header is calculated from the actual timestamp differences of the buffered frames, not the configured FPS constant.

---

### `square_wave.py` — GPIO Test Utility

Generates a 5600 Hz square wave (50% duty cycle) on GPIO pin 17 using `gpiozero.ToneBuzzer` / raw PWM. Used to verify the GPIO alert hardware is wired correctly before running the full application. Stop with Ctrl+C.

```bash
python src/moosedetector/square_wave.py
```

---

## Models

### `yolo26_best_v1.pt` (PyTorch — active)

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 / YOLO26 |
| Input resolution | 320 × 320 |
| Classes | `Animal` (0), `car` (1), `person` (2) |
| File size | ~5.3 MB |
| Framework | Ultralytics 8.4.14 + PyTorch 2.10 |
| Trained | Thermal animal detection dataset |

### `yolo26_ncnn_model_v1/` (NCNN — future)

Converted for the NCNN inference framework for faster edge deployment. The wrapper `model_ncnn.py` provides a compatible interface. Not used in the current `main` / `data_collection` branches — intended for the `feature-AI-HAT` branch with Hailo-8L acceleration.

---

## Configuration

All configuration is in [src/moosedetector/config.py](src/moosedetector/config.py). There is no external config file — edit the dataclass defaults directly.

Key values to change for a new deployment:

```python
# DetectionConfig
model_path = "/home/moose/projects/MooseDetector/models/yolo26_best_v1.pt"
confidence = 0.5           # Detection threshold (0–1)

# AlertConfig
gpio_pin = 17              # BCM pin number for alert output
persistence_frames = 30    # Frames to hold alert after object leaves (~3.3 s at 9 FPS)
alert_classes = ["Animal", "person"]

# DataCollectionConfig
button_pin = 23            # BCM pin for record button
led_pin = 24               # BCM pin for recording LED
pre_buffer_seconds = 10    # Seconds of footage held in RAM
fps = 9                    # Camera frame rate

# MetricsConfig
log_path = "/home/moose/projects/MooseDetector/logs"
```

---

## Installation

### 1. Install Seek Thermal SDK (C bindings)

```bash
# Download SDK v4.4.2.20 from https://developer.thermal.com/support/home
# Extract to:
~/projects/ThermalCameraSDK/

# Install SDL2 (required by SDK)
sudo apt-get install libsdl2-dev

# Install udev rules for USB device detection
cd ~/projects/ThermalCameraSDK/Seek_Thermal_SDK_4.4.2.20/aarch64-linux-gnu/
sudo cp driver/udev/10-seekthermal.rules /etc/udev/rules.d
sudo udevadm control --reload

# Add library path to shell environment
echo 'export LD_LIBRARY_PATH=~/projects/ThermalCameraSDK/Seek_Thermal_SDK_4.4.2.20/aarch64-linux-gnu/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Seek Thermal SDK (Python wrapper)

```bash
pip install -e ~/projects/ThermalCameraSDK/seekcamera-python
```

### 3. Create virtual environment and install dependencies

```bash
cd ~/projects/MooseDetector/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify

```bash
python -c "import seekcamera; print('Seek SDK loaded successfully')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Running the Application

```bash
cd ~/projects/MooseDetector/
source venv/bin/activate
python src/main.py
```

Stop with **Ctrl+C**. The application will flush any open video file and clean up GPIO before exiting.

### Quick GPIO test (before first run)

```bash
python src/moosedetector/square_wave.py
```

Confirm you hear / see the alert device respond, then press Ctrl+C.

---

## Deployment (Run on Boot)

### Option A — GNOME Autostart (desktop session)

Copy the desktop file to the autostart directory:

```bash
cp deploy/moosedetector.desktop ~/.config/autostart/
```

The `.desktop` file sets `LD_LIBRARY_PATH` and `PYTHONPATH` automatically and opens a terminal window so you can see log output.

### Option B — Shell script (SSH / headless)

```bash
scripts/run_on_boot.sh
```

This activates the venv and launches `src/main.py`. Wire it into `cron @reboot` or a systemd user service as needed.

---

## GPIO Pinout

All pin numbers are **BCM** (Broadcom) numbering.

| BCM Pin | Direction | Function |
|---------|-----------|---------|
| 17 | Output | Alert signal — HIGH when target detected |
| 23 | Input | Record button (active LOW via internal pull-up) |
| 24 | Output | Recording indicator LED |

---

## Outputs

### Video files (`videos/`)

Two AVI files are created per recording session:

| File | Content |
|------|---------|
| `raw_YYYYMMDD_HHMMSS.avi` | Unmodified thermal frames (RGB) |
| `overlay_YYYYMMDD_HHMMSS.avi` | Thermal frames with YOLO bounding boxes, track IDs, and metrics overlay |

Both files include the pre-buffer footage recorded before the button was pressed.

### Metrics logs (`logs/`)

One CSV per run, named `metrics_YYYYMMDD_HHMMSS.csv`:

| Column | Description |
|--------|-------------|
| `timestamp` | Unix time of frame |
| `fps` | Smoothed FPS (20-frame window) |
| `preprocess_ms` | YOLO preprocess time |
| `inference_ms` | YOLO inference time |
| `postprocess_ms` | YOLO postprocess time |
| `object_count` | Number of tracked objects in frame |
| `alert` | Boolean — alert pin state |

---

## Development Setup

### Static IP (Raspberry Pi OS)

```bash
sudo nano /etc/dhcpcd.conf
```

Add:
```
interface wlan0
static ip_address=192.168.5.44/24
static routers=192.168.5.1
static domain_name_servers=192.168.5.1 8.8.8.8
```

```bash
sudo reboot
```

### SSH

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

Connect from your development machine:
```bash
ssh moose@192.168.5.44
```

### VS Code Remote SSH

1. Install the **Remote – SSH** extension in VS Code.
2. Open Command Palette → `Remote-SSH: Open SSH configuration file`.
3. Add:
   ```
   Host moosedetector-pi
       HostName 192.168.5.44
       User moose
   ```
4. Connect via Command Palette → `Remote-SSH: Connect to Host` → `moosedetector-pi`.

This gives full access to the project directory, Python interpreter, debugger, and terminal over SSH.

### Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable release |
| `data_collection` | Active development — recording + metrics features |
| `feature-AI-HAT` | Experimental Hailo-8L NPU acceleration |
