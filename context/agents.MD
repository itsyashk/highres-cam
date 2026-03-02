# 🔬 Vimba Centroid Lab — Agent Codex

> **Sparky's Guide to the 850nm Vision Lab** ✨
>
> _A love letter from the codebase to whichever AI agent reads it next._

---

## 🌟 Project Overview — "What Does Sparky Even Do?"

This is a **high-precision sub-pixel centroid analysis system** for Allied Vision industrial cameras operating under 850nm illumination at 20.2 megapixels. It captures live video from GigE cameras, finds bright fiducials (blobs) in the image, and computes their centers to **sub-pixel accuracy** using isophote interpolation + algebraic circle fitting (Pratt fit + Levenberg–Marquardt refinement).

**Who is it for?**
Optical engineers, motion-capture researchers, and hardware calibration teams at Relativity who need to measure centroid jitter, calibrate pixel scales, and validate camera alignment — all in real-time with quantitative precision.

**Two interfaces, one brain:**

1. **Web UI** (primary, actively developed) — FastAPI + WebSocket video streaming with inline HTML/CSS/JS served from `web_backend.py`.
2. **Desktop GUI** (legacy/secondary) — PySide6 native window via `ui_main.py`, launched with `python -m vimba_centroid_lab`.

---

## 🧰 Tech Stack — "What's Under the Hood?"

### Languages

| Language     | Usage                                         |
| ------------ | --------------------------------------------- |
| Python 3.10+ | All backend, camera control, image processing |
| JavaScript   | Inline frontend in `web_backend.py`           |
| HTML/CSS     | Inline frontend (single-page app)             |
| YAML         | Configuration (`config.yaml`)                 |
| Bash         | Server startup script (`start_server.sh`)     |

### Core Libraries (from `requirements.txt`)

| Package             | Version      | Role                                               |
| ------------------- | ------------ | -------------------------------------------------- |
| `fastapi`           | ≥ 0.111      | REST API + WebSocket server                        |
| `uvicorn[standard]` | ≥ 0.30       | ASGI server                                        |
| `opencv-python`     | ≥ 4.9        | Image encoding, thresholding, connected components |
| `numpy`             | ≥ 1.24       | Array math, frame manipulation                     |
| `scipy`             | ≥ 1.11       | Non-linear least-squares circle fitting            |
| `PySide6`           | ≥ 6.7, < 6.8 | Desktop GUI (Qt bindings)                          |
| `PyYAML`            | ≥ 6.0        | Config file parsing                                |
| `matplotlib`        | ≥ 3.8        | Live centroid-delta plotting in desktop GUI        |
| `pytest`            | ≥ 7.4        | Unit testing                                       |

### External SDKs (not on PyPI)

| Package               | Source                      | Role                                 |
| --------------------- | --------------------------- | ------------------------------------ |
| `vimba` (VimbaPython) | Allied Vision SDK installer | Hardware camera control via GigE/USB |

### Infrastructure

- **OS**: Linux (udev rules required for camera access)
- **Network**: GigE Vision (cameras connected via Power-over-Ethernet)
- **License**: MIT

---

## 📂 Project Structure — "Where Does Sparky Keep Things?"

```
highres-cam/
├── vimba_centroid_lab/          # 🧠 Core application package
│   ├── __init__.py              #    Package marker
│   ├── __main__.py              #    Entry point for `python -m vimba_centroid_lab` (desktop GUI)
│   ├── main.py                  #    Desktop GUI launcher (loads config, creates QApplication)
│   ├── web_backend.py           #    🌐 FastAPI backend + ENTIRE inline frontend (2097 lines!)
│   ├── camera_vimba.py          #    📷 Camera abstraction: BaseCamera, MockCamera, VimbaCamera, CameraController
│   ├── processing.py            #    🧮 Image processing: blob detection, baseline & sub-pixel centroid
│   ├── viz.py                   #    🎨 Visualization helpers: centroid overlay, zoom ROI with grid
│   ├── ui_main.py               #    🖥️ PySide6 desktop GUI (MainWindow with sliders, zoom, plot)
│   ├── config.yaml              #    ⚙️ Default parameters (thresholds, ROI size, zoom scale, etc.)
│   ├── requirements.txt         #    📦 Python dependencies
│   ├── README.md                #    📖 Detailed setup & usage guide
│   ├── LICENSE                  #    MIT License
│   └── output/                  #    📁 Saved CSV series data (gitignored)
│
├── VimbaPython/                 #    📚 Allied Vision SDK Python bindings (vendored, do NOT modify)
├── Vimba_documentation/         #    📚 Vendor docs (gitignored)
│
├── tests/
│   ├── test_synthetic.py        #    ✅ Sub-pixel centroid accuracy test on synthetic disk
│   └── 1000photos/              #    📁 Burst-saved photos from camera
│
├── intrinsics/
│   └── cv.ipynb                 #    📓 Camera intrinsics calibration notebook
│
├── notebooks/
│   └── capture_roi.ipynb        #    📓 ROI capture exploration notebook
│
├── context/                     #    📄 Project documentation
│   ├── agents.MD                #    👈 YOU ARE HERE
│   ├── context.MD               #    Product context & domain model
│   └── project.MD               #    Project notes
│
├── test_camera.py               #    🧪 Quick single-camera hardware smoke test
├── test_multicam.py             #    🧪 Multi-camera WebSocket stream verifier
├── test_new_api.py              #    🧪 REST API endpoint validation script
├── start_server.sh              #    🚀 Server startup script (kills old processes, activates venv)
├── server.log                   #    📋 Last server run log
├── NEW_API_README.md            #    📖 Frontend-backend communication protocol docs
├── README.md                    #    📖 Top-level README (points to vimba_centroid_lab/)
└── .gitignore                   #    🚫 Ignores __pycache__, venvs, output/, Vimba_documentation/
```

---

## ⌨️ Key Commands — "How Does Sparky Run?"

### Start the Web Server (primary usage)

```bash
# Option 1: Via the startup script (recommended)
./start_server.sh

# Option 2: Manual uvicorn
source venv/bin/activate
python -m uvicorn vimba_centroid_lab.web_backend:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

### Start the Desktop GUI (legacy)

```bash
source .venv/bin/activate
python -m vimba_centroid_lab
```

### Run Tests

```bash
# Unit test (sub-pixel precision on synthetic disk)
pytest -q

# API endpoint tests (server must be running)
python test_new_api.py

# Quick camera hardware check
python test_camera.py

# Multi-camera WebSocket stream test (server must be running)
python test_multicam.py
```

### Install Dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r vimba_centroid_lab/requirements.txt
# VimbaPython must be installed separately via Allied Vision SDK
```

---

## 🔑 Environment Variables — "Sparky's Secrets"

This project does **not** use `.env` files or `os.environ` / `process.env` for configuration. All settings are managed through:

| Source                                  | Variables                                                                                                                                                                                                                            | Notes                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `config.yaml`                           | `pixel_size_mm_per_px`, `known_diameter_mm`, `threshold_core`, `threshold_non_black`, `number_of_rays`, `roi_size_px`, `output_dir`, `buffer_count`, `percentile_threshold`, `max_candidates`, `capture_series_length`, `zoom_scale` | **Required** — loaded at startup by `main.py` (desktop GUI). Web backend uses hardcoded defaults. |
| `camera_state` dict in `web_backend.py` | `exposure` (5000µs), `gain` (0dB), `frame_rate` (5.0 FPS), `pixel_size_mm` (0.6), `known_diameter_mm` (12.7), `series_target` (50), `save_target` (1000)                                                                             | Runtime state; adjustable via API/UI sliders                                                      |
| `web_backend.py` constants              | `PHOTO_SAVE_WORKERS` (auto: `min(8, max(2, cpu_count))`), `PHOTO_SAVE_QUEUE_MAX` (32)                                                                                                                                                | Tuning for burst photo saving                                                                     |
| `camera_vimba.py` constants             | `TARGET_BANDWIDTH` (31,000,000 bytes/s), `GevSCPD` (10000 ticks), `GevSCFTD` (100000)                                                                                                                                                | GigE camera network tuning                                                                        |
| Inline JS                               | `DEBOUNCE_DELAY` (300ms)                                                                                                                                                                                                             | Frontend slider debounce timing                                                                   |

> **Note**: The Vimba SDK itself requires system-level setup (udev rules, SDK installation path) but no env vars are referenced in the code.

---

## 🏗️ Architecture & Data Flow — "How Sparky Thinks"

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                        │
│  Inline HTML/CSS/JS served by GET /                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Camera Cards  │  │ Control      │  │ Zoom + Centroid      │   │
│  │ (img per cam) │  │ Panel        │  │ Analysis Panel       │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘   │
│         │ binary frames    │ JSON                                 │
│         │ via WS           │ via REST + WS                       │
└─────────┼──────────────────┼─────────────────────────────────────┘
          │                  │
    ┌─────▼──────┐   ┌──────▼──────┐
    │ /ws/{idx}  │   │ /ws/camera  │     WebSocket endpoints
    │ Video WS   │   │ State WS    │
    └─────┬──────┘   └──────┬──────┘
          │                  │
    ┌─────▼──────────────────▼──────┐
    │       FastAPI Backend          │
    │   (web_backend.py — 2097 LOC)  │
    │                                │
    │  ┌────────────────────────┐    │
    │  │  Camera Processing     │    │
    │  │  Threads (per camera)  │    │
    │  └─────────┬──────────────┘    │
    │            │                   │
    │  ┌─────────▼──────────────┐    │
    │  │  CameraController      │    │
    │  │  (camera_vimba.py)     │    │
    │  │  Time-multiplexed      │    │
    │  │  alternating capture   │    │
    │  └─────────┬──────────────┘    │
    │            │                   │
    │  ┌─────────▼──────────────┐    │
    │  │  VimbaCamera instances  │    │  ← or MockCamera if no hardware
    │  │  (GigE via VimbaPython) │    │
    │  └────────────────────────┘    │
    │                                │
    │  ┌────────────────────────┐    │
    │  │  processing.py         │    │  ← blob detection + sub-pixel centroid
    │  │  (Cam 0 only for CPU)  │    │
    │  └────────────────────────┘    │
    └────────────────────────────────┘
```

### Data Flow (Frame Lifecycle)

1. **Capture**: `CameraController._alternating_capture_loop()` cycles through cameras one-at-a-time. Each `VimbaCamera` starts → grabs one frame via `_on_frame` callback → puts it in its `Queue` → stops.
2. **Processing**: Per-camera background threads (`_run_camera_loop`) pull frames from the camera queue, store them in `camera_state["current_frames"]`, optionally run centroid analysis (Cam 0 only), and push to WebSocket queues.
3. **Photo Saving**: If burst mode is active, frames are also `put_nowait()` onto `photo_save_queue`, consumed by a pool of `PHOTO_SAVE_WORKERS` daemon threads that encode to JPEG and write to `tests/1000photos/`.
4. **Streaming**: `video_websocket_endpoint()` per camera pulls from the WebSocket queue, encodes to JPEG (quality 85), and sends binary messages to the browser.
5. **Display**: Browser `<img>` tags are updated via `URL.createObjectURL(new Blob([data]))` from the WebSocket binary messages.

### Key Patterns

- **Time-Multiplexed Capture**: Only ONE camera streams at a time to prevent GigE bandwidth saturation.
- **Queue-based Decoupling**: Camera threads → `Queue` → processing threads → `Queue` → WebSocket handlers. This prevents frame drops from blocking the camera.
- **Dual WebSocket Architecture**: Separate channels for binary video data (`/ws/{idx}`) and JSON state updates (`/ws/camera`).
- **Inline Frontend**: The entire SPA lives as a Python string in `web_backend.py`'s `index()` handler — no `npm`, no bundler, no `package.json`.

---

## 📐 Coding Conventions — "How Sparky Writes Code"

### Naming

- **Files**: `snake_case.py` (e.g., `camera_vimba.py`, `web_backend.py`)
- **Classes**: `PascalCase` (e.g., `CameraController`, `MockCamera`, `MainWindow`)
- **Functions/Methods**: `snake_case` with leading underscore for private (e.g., `_on_frame`, `_pump_frames`, `_alternating_capture_loop`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `VIMBA_AVAILABLE`, `TARGET_BANDWIDTH`, `PHOTO_SAVE_WORKERS`)
- **Config keys**: `snake_case` in YAML (e.g., `pixel_size_mm_per_px`, `threshold_core`)

### Type Annotations

- Used throughout with `from __future__ import annotations` for forward references.
- `Optional`, `List`, `Dict`, `Tuple` from `typing` module.
- Union types with `|` syntax (Python 3.10+): `dict[str, Any]`, `tuple[float, float] | None`.

### Imports

- Standard library first → third-party → local package (relative imports with `.`)
- Conditional imports for optional dependencies:
  ```python
  try:
      from vimba import Vimba, FrameStatus
      VIMBA_AVAILABLE = True
  except ImportError:
      VIMBA_AVAILABLE = False
  ```

### Error Handling

- Aggressive try/except with emoji-prefixed print logging: `✓`, `❌`, `⚠`, `📸`, `🔌`
- Camera SDK calls always wrapped in fallback chains (try enum API → try string API → graceful skip).
- HTTP errors use FastAPI's `HTTPException` with descriptive detail messages.

### Architecture Patterns

- **Facade pattern**: `CameraController` wraps multiple `BaseCamera` implementations.
- **Abstract base class**: `BaseCamera` defines the interface; `VimbaCamera` and `MockCamera` implement it.
- **Producer-consumer via `Queue`**: Threading architecture uses standard library `queue.Queue` objects throughout.
- **Global mutable state**: `camera_state` dict in `web_backend.py` acts as the source of truth for runtime state.

### Testing

- `pytest` for unit tests; synthetic data for hardware-free testing.
- Manual test scripts (`test_camera.py`, `test_multicam.py`, `test_new_api.py`) for integration.
- Accuracy assertion: sub-pixel centroid must be within `0.02 px` of ground truth.

---

## 🚫 What to Avoid — "Don't Touch Sparky's Stuff"

### Auto-Generated / Vendored (DO NOT MODIFY)

| Path                            | Reason                                     |
| ------------------------------- | ------------------------------------------ |
| `VimbaPython/`                  | Vendored Allied Vision SDK Python bindings |
| `Vimba_documentation/`          | Vendor documentation (gitignored)          |
| `__pycache__/`                  | Bytecode cache                             |
| `.venv/`, `venv/`, `vimba_env/` | Virtual environments                       |
| `vimba_centroid_lab/output/`    | Generated CSV output data (gitignored)     |
| `tests/1000photos/`             | Burst-captured photos                      |
| `.pytest_cache/`                | Test cache                                 |
| `server.log`                    | Runtime log file                           |

### Structural Warnings

- **Do NOT create a separate frontend project** (no React, no Vue, no `package.json`). The entire UI lives inline in `web_backend.py`.
- **Do NOT install VimbaPython via pip** — it must come from the Allied Vision SDK installer.
- **Do NOT start multiple Vimba contexts** — only `CameraController` should manage the single `Vimba.get_instance()`.
- **Do NOT stream multiple cameras simultaneously** over GigE — the time-multiplexed design exists to prevent packet collisions.

---

## 🤖 Agent Instructions — "Dear Future Agent, With Love, Sparky"

### 🔨 How to Implement Features

1. **Camera features** → Modify `camera_vimba.py`. Add methods to `BaseCamera` (interface), then implement in both `VimbaCamera` and `MockCamera` to maintain testability.
2. **Image processing algorithms** → Add to `processing.py`. Keep functions pure (input: `np.ndarray`, output: tuples/arrays). Keep them compatible with Mono8 (single-channel uint8).
3. **New API endpoints** → Add to `web_backend.py` using FastAPI decorators. Follow the existing pattern: validation → camera_state update → WebSocket broadcast.
4. **UI changes** → Edit the HTML/CSS/JS string inside `web_backend.py`'s `index()` function. There is no hot-reload for this — you must restart the server.
5. **Desktop GUI changes** → Edit `ui_main.py`. This is a secondary interface; check if the user is actively using it before investing time here.

### 🔧 How to Refactor Safely

1. **Always run `pytest -q`** after changes to `processing.py` — the sub-pixel test has a tight 0.02 px tolerance.
2. **Test with MockCamera first** — if Vimba SDK is not available, `CameraController` will report zero cameras. The mock camera is ideal for UI/frontend development.
3. **Watch the thread boundaries** — the backend runs camera loops, photo-save workers, and WebSocket handlers on separate threads. Never do blocking I/O on the main asyncio loop.
4. **When modifying `camera_state`** — this is a plain dict shared across threads. For simple reads/writes of scalars it's okay, but if you add lists or nested structures, consider `threading.Lock`.
5. **Frontend changes are risky** — the inline HTML/CSS/JS block is ~1300 lines embedded in a Python string. Use careful string escaping. Test by refreshing the browser after restart.

### 🐛 How to Debug

1. **Camera not connecting**: Run `python test_camera.py` to check basic Vimba connectivity. Verify udev rules and GigE interface.
2. **Incomplete frames**: Check GigE bandwidth settings in `camera_vimba.py` — `TARGET_BANDWIDTH`, `GevSCPD`, `GevSCFTD`. These are tuned for dual-camera operation.
3. **WebSocket not streaming**: Check `test_multicam.py`. Verify the camera count from `/api/camera/status`. The frontend dynamically creates `<img>` tags based on camera count.
4. **Processing errors**: Enable the debug prints in `_run_camera_loop` (they're partially commented out). Check that frames are actually arriving in the queues.
5. **Server won't start**: Run `start_server.sh` — it kills existing uvicorn/vimba processes first. Check `server.log`.

### ⚠️ When to Ask Before Acting

- **Before changing the time-multiplexed capture logic** (risk of GigE packet collisions).
- **Before modifying the WebSocket message format** (frontend and backend are tightly coupled).
- **Before adding new Python dependencies** (the Vimba SDK environment can be fragile).
- **Before changing exposure/gain validation ranges** (hardware-specific limits).
- **Before restructuring `camera_state`** (shared across threads with no locking).
- **Before touching `VimbaPython/`** (vendored SDK code — not ours to modify).

---

> _"Measure twice, centroid once."_ — Sparky 🔬✨
