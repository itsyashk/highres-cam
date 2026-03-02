"""
Wrapper around Allied Vision VimbaPython camera providing threaded frame streaming.

If Vimba is not available or no camera is detected, falls back to MockCamera that
generates synthetic blurred disks useful for unit testing and UI development.

Primary class: CameraController
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Optional, List

import numpy as np

try:
    from vimba import Vimba, FrameStatus  # type: ignore
    VIMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    VIMBA_AVAILABLE = False


class BaseCamera:
    """Abstract base camera interface."""

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def set_exposure(self, us: float):
        """Set exposure time in micro-seconds."""

    def set_gain(self, db: float):
        """Set analog gain in decibels."""

    def set_frame_rate(self, fps: float):
        """Target acquisition frame-rate in frames per second."""


class MockCamera(BaseCamera):
    """Synthetic camera for development when no hardware is present."""

    def __init__(self, frame_queue: Queue, width: int = 5472, height: int = 3648, period: float = 0.033):
        self.q = frame_queue
        self.width = width
        self.height = height
        self.period = period
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t0 = time.time()

    def _generate_frame(self, t: float) -> np.ndarray:
        """Generate a blurred bright disk with slight motion for realism."""
        # Offset motion based on queue ID to differentiate cameras
        offset_x = (id(self.q) % 100) / 100.0
        x0 = int(self.width * (0.3 + 0.3 * np.sin(t * 0.7 + offset_x * 6.28)))
        y0 = int(self.height * (0.3 + 0.3 * np.cos(t * 1.1)))
        yy = np.arange(self.height, dtype=np.float32)[:, np.newaxis]
        xx = np.arange(self.width, dtype=np.float32)[np.newaxis, :]
        r = 120  # px
        dist = np.hypot(xx - x0, yy - y0)
        img = np.clip(255 * np.exp(-((dist) ** 2) / (2 * (r ** 2))), 0, 255).astype(np.float32)
        noise = np.random.normal(0, 3, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img

    def _loop(self):
        while not self._stop.is_set():
            t = time.time() - self.t0
            frame = self._generate_frame(t)
            if not self.q.full():
                self.q.put(frame)
            time.sleep(self.period)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # Dummy setters – keep API compatible
    def set_exposure(self, us):
        pass

    def set_gain(self, db):
        pass

    def set_frame_rate(self, fps):
        self.period = 1.0 / fps


class VimbaCamera(BaseCamera):
    """Real Allied Vision camera via VimbaPython SDK."""

    def __init__(self, frame_queue: Queue, camera_id: Optional[str] = None, buffer_count: int = 32):
        if not VIMBA_AVAILABLE:
            raise RuntimeError("VimbaPython not available – install Allied Vision SDK")
        self.q = frame_queue
        self.buffer_count = buffer_count
        self._cam = None
        self._vimba_ctx = None  # holds the Vimba instance context manager
        self._running = False
        self.camera_id = camera_id
        self._gvsp_adjusted = False   # only run packet size negotiation once
        self._frame_seq = 0            # monotonic frame counter for pipeline tracing
        self._cam_open = False         # True after __enter__() — camera stays open between streaming cycles
        self._frame_event = threading.Event()  # set by _on_frame so the loop doesn't race the processing thread
        self._open_failed = False      # True if open permanently failed — prevents retry spam

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------
    def _on_frame(self, cam, frame):  # type: ignore
        try:
            status = frame.get_status()
            if status == FrameStatus.Complete:
                img = frame.as_numpy_ndarray()
                self._frame_seq += 1
                # Drain stale frames so only the latest is ever queued
                drained = 0
                while not self.q.empty():
                    try:
                        self.q.get_nowait()
                        drained += 1
                    except Exception:
                        break
                self.q.put(img.copy())
                self._frame_event.set()  # signal alternating loop without competing with processing thread
                # TRACE: frame arrived from Vimba hardware into camera queue
                # print(f"[PIPE|{self.camera_id[-6:]}|1_VIMBA→CAM_Q] seq={self._frame_seq} t={time.time():.3f} drained_stale={drained} q_size=1")
            else:
                # Log incomplete frames (throttled)
                if not hasattr(self, '_incomplete_count'):
                    self._incomplete_count = 0
                self._incomplete_count += 1
                if self._incomplete_count <= 3 or self._incomplete_count % 500 == 0:
                    print(f"WARNING: [{self.camera_id}] incomplete frames: {self._incomplete_count} (last status={status})")

            # CRITICAL: Always recycle the frame to keep streaming alive
            cam.queue_frame(frame)

        except Exception as e:
            print(f"ERROR: Frame handler for {self.camera_id}: {e}")
            try:
                cam.queue_frame(frame)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open_camera(self):
        """Open camera context and apply one-time configuration.

        Called ONCE before the alternating loop starts. Keeps the camera open
        between streaming cycles so start()/stop() only toggle streaming (~10ms)
        instead of doing a full open/configure/close (~850ms) every cycle.
        """
        if self._cam_open:
            return  # already open, nothing to do

        if self._open_failed:
            raise RuntimeError(f"Camera {self.camera_id} previously failed to open — skipping")

        if not self._cam:
            raise RuntimeError("Camera not initialized. Controller must assign a camera instance.")

        from vimba import AccessMode  # type: ignore
        t0 = time.time()

        # set_access_mode() must be called BEFORE __enter__().
        # If it raises "inside of 'with'" the camera is already open at the SDK level
        # (left in that state by a previous failed __enter__() call) — treat as open.
        try:
            self._cam.set_access_mode(AccessMode.Full)
        except Exception as e:
            if 'inside of' in str(e):
                # Camera already entered at SDK level despite _cam_open=False.
                # This happens when a previous __enter__() threw partway through.
                # Mark open and continue to streaming without re-entering.
                print(f"WARNING: {self.camera_id} already in SDK 'with' scope — treating as open")
                self._cam_open = True
            else:
                print(f"ERROR: Failed to open camera {self.camera_id}: {e}")
                self._open_failed = True
                raise

        if not self._cam_open:
            try:
                self._cam.__enter__()
                self._cam_open = True
            except Exception as e:
                # __enter__() may leave camera partially open at C++ level.
                # Try to exit cleanly, then give up on this camera.
                try:
                    self._cam.__exit__(None, None, None)
                except Exception:
                    pass
                print(f"ERROR: Failed to open camera {self.camera_id}: {e}")
                self._open_failed = True
                raise

        # GigE packet size negotiation — one-time, MTU doesn't change
        try:
            self._cam.GVSPAdjustPacketSize.run()
            while not self._cam.GVSPAdjustPacketSize.is_done():
                pass
            self._gvsp_adjusted = True
        except Exception:
            self._gvsp_adjusted = True

        # Bandwidth limit for multi-camera GigE (31 MB/s)
        TARGET_BANDWIDTH = 31_000_000
        try:
            try:
                self._cam.get_feature_by_name("StreamBytesPerSecond").set(TARGET_BANDWIDTH)
            except Exception:
                self._cam.get_feature_by_name("DeviceLinkThroughputLimit").set(TARGET_BANDWIDTH)
        except Exception as e:
            print(f"WARNING: Could not set bandwidth limit for {self.camera_id}: {e}")

        # Inter-packet delay for GigE stability
        try:
            self._cam.get_feature_by_name("GevSCPD").set(10000)
        except Exception:
            pass
        try:
            self._cam.get_feature_by_name("GevSCFTD").set(100000)
        except Exception:
            pass

        # Pixel format
        try:
            from vimba import PixelFormat as PF  # type: ignore
            try:
                self._cam.set_pixel_format(PF.Mono8)
            except Exception:
                try:
                    feat = self._cam.get_feature_by_name('PixelFormat')
                    try:
                        feat.set(PF.Mono8)
                    except Exception:
                        feat.set('Mono8')
                except Exception as e2:
                    print(f"WARNING: Could not set Mono8 on {self.camera_id}: {e2}")
        except Exception as e:
            print(f"WARNING: Could not enforce Mono8 pixel format on {self.camera_id}: {e}")

        # print(f"[PIPE|{self.camera_id[-6:]}|OPEN] Camera opened and configured in {time.time()-t0:.3f}s")

    def start(self):
        """Start streaming only — camera must already be open via open_camera()."""
        if self._open_failed:
            raise RuntimeError(f"Camera {self.camera_id} is unavailable")
        if not self._cam_open:
            # Fallback: open on first call if open_camera() wasn't called beforehand
            self.open_camera()
        self._frame_event.clear()
        self._cam.start_streaming(handler=self._on_frame, buffer_count=self.buffer_count)
        self._running = True
        print(f"Camera {self.camera_id} streaming started")

    def attach_camera(self, cam_obj):
        """Attaches a Vimba Camera object found by the controller."""
        self._cam = cam_obj
        self.camera_id = cam_obj.get_id()

    def stop(self):
        """Stop streaming only — camera stays open for the next cycle."""
        if self._cam and self._running:
            try:
                self._cam.stop_streaming()
            except Exception:
                pass
            self._running = False

    def close(self):
        """Close camera context — called at shutdown only."""
        if self._running:
            try:
                self._cam.stop_streaming()
            except Exception:
                pass
            self._running = False
        if self._cam and self._cam_open:
            try:
                self._cam.__exit__(None, None, None)
            except Exception:
                pass
            self._cam_open = False

    # Detached context management from VimbaCamera to CameraController

    def set_exposure(self, exposure_us: float) -> None:
        """Set camera exposure time in microseconds."""
        if hasattr(self, '_cam') and self._cam:
            try:
                self._cam.ExposureTime.set(exposure_us)
            except Exception as e:
                print(f"ERROR: [{self.camera_id}] set_exposure failed: {e}")
                try:
                    self._cam.get_feature_by_name('ExposureTime').set(exposure_us)
                except Exception as e2:
                    print(f"ERROR: [{self.camera_id}] set_exposure fallback failed: {e2}")

    def set_gain(self, gain_db: float) -> None:
        """Set camera gain in dB."""
        if hasattr(self, '_cam') and self._cam:
            try:
                self._cam.Gain.set(gain_db)
            except Exception as e:
                print(f"ERROR: [{self.camera_id}] set_gain failed: {e}")
                try:
                    self._cam.get_feature_by_name('Gain').set(gain_db)
                except Exception as e2:
                    print(f"ERROR: [{self.camera_id}] set_gain fallback failed: {e2}")

    def set_frame_rate(self, fps: float):
        if self._cam:
            self._cam.AcquisitionFrameRate.set(fps)

    def capture_single_frame(self, timeout_ms: int = 2000):
        """Capture a single frame synchronously (for time-multiplexed mode).

        Returns the frame as numpy array or None if failed.
        """
        if not self._cam:
            print(f"ERROR: Camera {self.camera_id} not available for single capture")
            return None

        try:
            frame = self._cam.get_frame(timeout_ms=timeout_ms)
            if frame.get_status() == FrameStatus.Complete:
                return frame.as_numpy_ndarray().copy()
            else:
                print(f"WARNING: [{self.camera_id}] single capture incomplete frame: status={frame.get_status()}")
                return None
        except Exception as e:
            print(f"ERROR: [{self.camera_id}] single frame capture: {e}")
            return None

    def open_for_capture(self):
        """Open camera for single-frame capture mode (not streaming)."""
        if not self._cam:
            print(f"ERROR: No camera attached to open_for_capture")
            return False
        if self._cam_open:
            return True  # already open from multiplexed mode
        try:
            from vimba import AccessMode
            self._cam.set_access_mode(AccessMode.Full)
            self._cam.__enter__()
            self._cam_open = True
            try:
                from vimba import PixelFormat as PF
                self._cam.set_pixel_format(PF.Mono8)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"ERROR: Failed to open camera {self.camera_id} for capture: {e}")
            return False

    def close_capture(self):
        """Close camera after single-frame capture session (no-op if multiplexed mode owns it)."""
        if self._cam and self._cam_open and not self._running:
            try:
                self._cam.__exit__(None, None, None)
            except Exception as e:
                print(f"WARNING: Error closing camera {self.camera_id}: {e}")
            self._cam_open = False


class CameraController:
    """Unified facade managing multiple cameras (real or mock)."""

    def __init__(self):
        # We process cameras into a list of (camera_instance, queue)
        self.cameras: List[BaseCamera] = []
        self.queues: List[Queue] = []
        self._vimba_ctx = None
        
        if VIMBA_AVAILABLE:
            from vimba import Vimba
            self.Vimba = Vimba
        else:
            self.Vimba = None

    def start(self):
        """Discover and start all connected cameras."""
        if self.Vimba:
            try:
                self._vimba_ctx = self.Vimba.get_instance()
                self._vimba_ctx.__enter__()

                cams_found = self._vimba_ctx.get_all_cameras()
                for cam_obj in cams_found:
                    q = Queue(maxsize=30)
                    vcam = VimbaCamera(q)
                    vcam.attach_camera(cam_obj)
                    self.cameras.append(vcam)
                    self.queues.append(q)

            except Exception as e:
                print(f"ERROR: Failed to initialize Vimba: {e}")
                if self._vimba_ctx:
                    self._vimba_ctx.__exit__(None, None, None)
                    self._vimba_ctx = None

        if len(self.cameras) == 0:
            print("ERROR: No cameras detected. Check connections.")
            return

        # Start cameras with staggered delays to prevent simultaneous stream bursts
        import time
        for i, cam in enumerate(self.cameras):
            cam.start()
            if i < len(self.cameras) - 1:
                time.sleep(3)

        print(f"Camera controller started: {len(self.cameras)} camera(s)")


    def stop(self):
        """Stop all cameras."""
        for cam in self.cameras:
            cam.stop()
        
        if self._vimba_ctx:
            self._vimba_ctx.__exit__(None, None, None)
            self._vimba_ctx = None

    # Global Setters (apply to all cameras)
    def set_exposure(self, us):
        for cam in self.cameras:
            cam.set_exposure(us)

    def set_gain(self, db):
        for cam in self.cameras:
            cam.set_gain(db)

    def set_frame_rate(self, fps):
        for cam in self.cameras:
            cam.set_frame_rate(fps)
    
    def get_queues(self) -> List[Queue]:
        return self.queues

    def start_multiplexed(self, interval_seconds: float = 0.333):
        """Start TRUE time-multiplexed capture mode.

        Only ONE camera is ever active at a time:
        - t=0.000: Start Camera 0, capture frame, stop
        - t=0.167: Start Camera 1, capture frame, stop
        - t=0.333: Start Camera 0, capture frame, stop
        - etc.
        
        Args:
            interval_seconds: Total cycle time for one full round of all cameras.
        """
        if self.Vimba and len(self.cameras) == 0:
            try:
                self._vimba_ctx = self.Vimba.get_instance()
                self._vimba_ctx.__enter__()

                cams_found = self._vimba_ctx.get_all_cameras()
                for cam_obj in cams_found:
                    q = Queue(maxsize=30)
                    vcam = VimbaCamera(q)
                    vcam.attach_camera(cam_obj)
                    self.cameras.append(vcam)
                    self.queues.append(q)

            except Exception as e:
                print(f"ERROR: Failed to discover cameras: {e}")
                if self._vimba_ctx:
                    self._vimba_ctx.__exit__(None, None, None)
                    self._vimba_ctx = None
                return

        if len(self.cameras) == 0:
            print("ERROR: No cameras available for multiplexed capture.")
            return

        # Pre-open all cameras ONCE so the alternating loop only toggles streaming
        # (eliminates the ~850ms open_took per cycle seen in traces)
        print(f"Pre-opening {len(self.cameras)} camera(s)...")
        available = []
        for cam in self.cameras:
            try:
                cam.open_camera()
                available.append(cam)
            except Exception as e:
                print(f"ERROR: Could not pre-open {cam.camera_id}, skipping: {e}")

        if not available:
            print("ERROR: No cameras successfully opened.")
            return

        num_cameras = len(self.cameras)
        per_camera_interval = interval_seconds / num_cameras

        self._multiplex_stop = threading.Event()
        self._multiplex_thread = threading.Thread(
            target=self._alternating_capture_loop,
            args=(per_camera_interval,),
            daemon=True
        )
        self._multiplex_thread.start()
        print(f"Multiplexed capture started: {num_cameras} camera(s), {per_camera_interval:.3f}s per camera")
    
    def _alternating_capture_loop(self, capture_interval: float):
        """Pipelined alternating capture.

        Key insight from traces:
          - stream_start_took: ~50ms
          - time-to-first-frame: ~440ms  (camera hardware — 2 frame periods at 4.5fps)
          - stop_streaming() blocks: ~270ms (Vimba waits for in-flight frame to finish)

        Old sequential:
          [start 50ms][exposure 440ms][stop 270ms] | [start 50ms][exposure 440ms][stop 270ms]
          Total cycle: 1520ms

        New pipelined: after getting the frame event, start the next camera FIRST,
        then stop the current one. The 270ms stop overlaps with the next camera's
        440ms exposure warmup instead of adding to it.
          [start 50ms][exposure 440ms][stop 270ms]
                                      [start 50ms][exposure 440ms][stop 270ms]
          Total cycle: ~980ms (~35% reduction)

        Both cameras stream simultaneously for ~270ms. Bandwidth stays safe:
        2 x 31MB/s = 62MB/s < 125MB/s GigE capacity.
        """
        import time

        num_cameras = len(self.cameras)
        current_camera = 0

        # Pre-start the first camera before entering the loop so the first
        # iteration can go straight to waiting for the frame event
        first_cam = self.cameras[0]
        first_cam._frame_event.clear()
        try:
            t0 = time.time()
            first_cam.start()
            # print(f"[PIPE|cam0|2_STREAM_START] t={time.time():.3f} pre-loop start_took={time.time()-t0:.3f}s")
        except Exception as e:
            print(f"[PIPE|cam0|2_ERROR] Failed pre-loop start: {e}")

        while not self._multiplex_stop.is_set():
            cam = self.cameras[current_camera]
            next_idx = (current_camera + 1) % num_cameras
            next_cam = self.cameras[next_idx]

            # Wait for current camera's frame via event.
            # _frame_event is set in _on_frame(), independent of the processing
            # thread that consumes from the queue — no race condition.
            t_wait = time.time()
            frame_received = cam._frame_event.wait(timeout=0.5)
            t_frame = time.time()

            if frame_received:
                pass  # print(f"[PIPE|cam{current_camera}|2_FRAME_EVENT] t={t_frame:.3f} waited={t_frame-t_wait:.3f}s")
            else:
                print(f"WARNING: cam{current_camera} no frame in 0.5s timeout")

            if next_idx != current_camera:
                # PIPELINE (N≥2): start next camera BEFORE stopping current.
                # next camera's 440ms exposure warmup overlaps current camera's
                # 270ms stop_streaming() — saves 270ms per switch vs sequential.
                # Per-iteration cost settles to: 170ms wait + 50ms start + 270ms stop = 490ms
                # Total cycle = N × 490ms for any N≥2.
                next_cam._frame_event.clear()
                try:
                    next_cam.start()
                except Exception as e:
                    print(f"ERROR: cam{next_idx} failed to start: {e}")

                try:
                    cam.stop()
                except Exception as e:
                    print(f"ERROR: cam{current_camera} failed to stop: {e}")
            else:
                # N=1: only one camera — stop then restart (pipeline doesn't apply,
                # next_idx == current_camera so we can't start before stopping)
                try:
                    cam.stop()
                except Exception as e:
                    print(f"ERROR: cam{current_camera} failed to stop: {e}")
                cam._frame_event.clear()
                try:
                    cam.start()
                except Exception as e:
                    print(f"ERROR: cam{current_camera} failed to restart: {e}")

            current_camera = next_idx
    
    def stop_multiplexed(self):
        """Stop time-multiplexed capture mode."""
        if hasattr(self, '_multiplex_stop'):
            self._multiplex_stop.set()
        if hasattr(self, '_multiplex_thread') and self._multiplex_thread.is_alive():
            self._multiplex_thread.join(timeout=2.0)

        # Close all cameras (stop streaming + close context)
        for cam in self.cameras:
            try:
                cam.close()
            except Exception:
                pass