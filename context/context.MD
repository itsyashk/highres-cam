# Product Context

**What problem does this app solve?**
The High-Resolution Camera Centroid Lab (Vimba Centroid Lab) provides a high-precision interface to evaluate the centroid precision of bright fiducials (specifically under 850nm illumination at 20.2MP resolution, according to the web UI). It is designed to capture, display, and analyze high-resolution images to find sub-pixel centroids of bright objects.

**Target User & Core Value Proposition:**
The target users are likely optical engineers, computer vision researchers, or hardware calibration engineers. The core value proposition is the ability to do live sub-pixel centroid analysis (using mathematical fits like the Pratt algorithm) with low-latency visual feedback on zoomed regions of interest (ROI), specifically over high-bandwidth GigE industrial cameras from Allied Vision (using the Vimba SDK).

# Current State

**What is built and working:**

- A dual-interface application: it provides both a robust modern web interface (via FastAPI and WebSockets serving HTML/JS) and a native desktop UI (`ui_main.py` using PySide6).
- A hardware-abstracted camera backend (`camera_vimba.py`) supporting real Allied Vision cameras (`VimbaCamera`) and falling back to synthetic data (`MockCamera`) for development without hardware.
- Real-time video streaming over WebSockets, supporting multiple connected cameras.
- Time-multiplexed multi-camera captures (fetching one frame per camera sequentially to prevent GigE network collisions).
- Mathematical sub-pixel centroid analysis (`processing.py`) utilizing coarse thresholding and refined circle fitting on isophote edges.
- Burst photo-saving capabilities to an `output/` or `tests/1000photos/` directory, run via background threading.

**Incomplete/Stubbed/Needs Fixes:**

- There is a known hack/TODO in `web_backend.py` around the global photo counter sharing during photo bursts across multiple cameras: `"Hack: we rely on the main camera to decrement the global counter?"`.
- Lack of explicit graceful shutdown for background camera threads.
- `camera_vimba.py` notes that `self._vimba_ctx` must be strictly managed by the Controller to share the Vimba instance properly across cameras.

# Domain Model

**Core Entities and Concepts:**

- `BaseCamera`: Abstract interface defining `start`, `stop`, `set_exposure`, `set_gain`, `set_frame_rate`.
- `VimbaCamera` / `MockCamera`: The concrete implementations. `MockCamera` generates moving blurred disks for testing without hardware.
- `CameraController`: A unified facade that manages multiple camera instances and handles the "time-multiplexed" Alternating Capture loop across them.
- `ConnectionManager`: Manages WebSocket connections for both the Video stream (`/ws`) and State updates (`/ws/camera`).
- `Blob`: Computed in `processing.py`, representing a bright spot in the thresholded mask characterized by its bounding box, centroid, area, and score.
- `Camera State`: A global Python dictionary in the backend dictating current system parameters (exposure, gain, fps, target saving counts, selected roi).

# Key Decision Log

**Architectural Decisions:**

- **Dual WebSocket Architecture (from `NEW_API_README.md`)**: Replaced "cooked" or simple polling communication with a real-time system divided into a Video socket and a State socket to separate concerns and prevent UI stutter.
- **Time-Multiplexed Capture (`CameraController._alternating_capture_loop`)**: Rather than reading all cameras concurrently, the system sequentially acquires single frames from each camera. This is an explicit decision to prevent Gigabit Ethernet (GigE) packet collisions and "Incomplete Frame" errors, which are common when multiple high-res cameras burst simultaneously.
- **Inline Frontend**: The entire frontend HTML/CSS/JS is served as a hardcoded string returned by the `index()` root endpoint in `web_backend.py`, instead of maintaining a separate React/Vue project. This favors single-file deployments tightly coupled to the backend logic.
- **Pratt Circle Fit**: The sub-pixel refinement specifically utilizes an algebraic Pratt fit combined with optional non-linear least squares (scipy LM), ensuring extremely robust sub-pixel edge detection over simple intensity-weighted bounding boxes.

# Boundaries & Constraints

**Known Limits & Constraints:**

- **Network Bandwidth**: High-resolution frames over GigE will hit bandwidth caps quickly. This necessitates the time-multiplexed capture approach.
- **Vimba Single Ownership**: The Vimba SDK allows only one central context. Multiple camera objects must share this context without violating scope (evident by the `NOTE: self._vimba_ctx must be managed by the Controller` comment).
- **Global Interpreter Lock (GIL) / CPU Bound**: Image processing (finding blobs, Scipy optimizations) runs on the CPU. The backend restricts centroid analysis strictly to "Camera 0" (`cam_idx == 0` check in `_run_camera_loop`) to save CPU cycles and prevent the video stream from dropping frames.
- **OS Constraints**: The Vimba SDK is tightly bound to Linux (requires explicit udev rules and python wheel installation outside of PyPI).

# Integration Map

**External Systems & APIs:**

- **Allied Vision Vimba SDK (`vimba` python package)**: The deepest hardware integration. Critical for capturing absolute frame data.
- **FastAPI / Uvicorn**: Serves the REST API and WebSockets.
- **OpenCV (`cv2`) & Scipy**: Core dependencies for matrix math, binary thresholding (`connectedComponentsWithStats`), bilinear sampling, and least-squares optimizations.
- **PySide6**: Used only in `ui_main.py` if the native desktop mode is spun up.

# Open Questions

1. **Concurrency during burst saving**: In `web_backend.py`, the photo save worker relies on Cam 0 decreasing the `save_remaining` target count for all cameras, but frames from other cameras might get dropped or miscounted if the threading causes a race condition.
2. **Graceful Shutdowns**: Currently, daemon threads are terminated abruptly with the Python process. Are there scenarios where half-saved CSVs or locked Vimba contexts cause issues on reboot?
3. **UI Syncing**: Is `ui_main.py` being fully deprecated in favor of the newer WebSocket-based web backend, or are both actively maintained?
4. **Vimba SDK Scope Errors**: History indicates bugs related to "Called 'Camera.set_access_mode()' inside of 'with' statement scope." The architecture might still be slightly fragile if cameras disconnect and reconnect.

# Agent Guidance

**What to keep in mind when working on this codebase:**

1. **Do not break the Time-Multiplexing**: The `CameraController` purposefully avoids grabbing frames simultaneously from multiple cameras. If asked to speed up multi-camera captures, beware of triggering GigE packet collisions that cause "Incomplete frame -1" errors.
2. **Watch your threads**: The backend heavily utilizes background Python threads to pump `queue.Queue` objects for WebSocket broadcasting and photo saving. Avoid blocking these loops with synchronous sleep calls or heavy CPU-bound logic (which is why image processing is locked to Cam 0).
3. **Respect the API boundary**: There is a documented REST API (`/api/camera/status`, `/api/camera/exposure`, `/api/camera/parameters`) and WebSocket spec. Do not change payload structures without updating the inline JS in `web_backend.py`'s `index()` handler.
4. **Mock Camera is your friend**: If you are writing tests or adding UI features, standard local runs will default to `MockCamera`. If modifying `MockCamera`, ensure it still returns the same shape (or `np.ndarray` structure) expected by OpenCV in `processing.py`.
5. **No Node/NPM frontend**: Do not look for `package.json` or `App.tsx`. The entire web interface is natively injected via the `index()` return string in `web_backend.py`. Modify HTML/JS/CSS directly there.
