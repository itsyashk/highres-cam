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
from typing import Optional

import numpy as np

try:
    from vimba import Vimba  # type: ignore
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
        x0 = int(self.width * (0.3 + 0.3 * np.sin(t * 0.7)))
        y0 = int(self.height * (0.3 + 0.3 * np.cos(t * 1.1)))
        yy, xx = np.indices((self.height, self.width))
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

    def __init__(self, frame_queue: Queue, camera_id: Optional[str] = None, buffer_count: int = 8):
        if not VIMBA_AVAILABLE:
            raise RuntimeError("VimbaPython not available – install Allied Vision SDK")
        self.q = frame_queue
        self.buffer_count = buffer_count
        self._cam = None
        self._vimba_ctx = None  # holds the Vimba instance context manager
        self._running = False
        self.camera_id = camera_id

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------
    def _on_frame(self, cam, frame):  # type: ignore
        try:
            img = frame.as_numpy_ndarray()
            #print(f"📸 Frame received from camera: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
            
            if not self.q.full():
                self.q.put(img.copy())
                #print(f"✓ Frame queued (queue size: {self.q.qsize()})")
            else:
                print(f"⚠ Frame queue full, dropping frame")
                
            # CRITICAL: Always queue the frame back to the camera to keep streaming alive
            cam.queue_frame(frame)
            #print(f"✓ Frame recycled back to camera")
            
        except Exception as e:
            print(f"❌ Frame handler error: {e}")
            import traceback
            traceback.print_exc()
            # Even on error, try to recycle the frame
            try:
                cam.queue_frame(frame)
               # print(f"✓ Frame recycled back to camera (error recovery)")
            except:
                print(f"❌ Failed to recycle frame after error")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        from vimba import Vimba  # local import to guard availability

        print("=== VIMBA CAMERA STARTUP ===")
        
        # Keep instance alive until stop() is called
        print("Getting Vimba instance...")
        self._vimba_ctx = Vimba.get_instance()
        self._vimba_ctx.__enter__()
        print("✓ Vimba instance acquired")

        vimba = self._vimba_ctx
        print("Discovering cameras...")
        cams = vimba.get_all_cameras()
        print(f"✓ Found {len(cams)} camera(s)")
        
        if not cams:
            raise RuntimeError("No Allied Vision cameras detected")
            
        if self.camera_id:
            print(f"Looking for specific camera ID: {self.camera_id}")
            cam = next((c for c in cams if c.get_id() == self.camera_id), cams[0])
            print(f"✓ Selected camera: {cam.get_id()}")
        else:
            cam = cams[0]
            print(f"✓ Using first available camera: {cam.get_id()}")
            
        self._cam = cam
        
        # Enter camera context (opens the camera)
        print("Opening camera (AccessMode.Full)...")
        from vimba import AccessMode  # type: ignore
        try:
            cam.open(access_mode=AccessMode.Full)
        except Exception as e:
            print(f"❌ Failed to open camera with AccessMode.Full: {e}")
            # Fallback to default __enter__ as last resort
            try:
                cam.__enter__()
            except Exception as e2:
                print(f"❌ Fallback camera open failed: {e2}")
                raise
        print("✓ Camera opened successfully")

        # Adjust packet size for GigE, ignore errors
        print("Attempting to adjust GigE packet size...")
        try:
            cam.GVSPAdjustPacketSize.run()
            while not cam.GVSPAdjustPacketSize.is_done():
                pass
            print("✓ GigE packet size adjusted")
        except Exception as e:
            print(f"⚠ GigE packet size adjustment failed (not a GigE camera?): {e}")

        # Ensure Mono8 pixel format
        print("Setting pixel format to Mono8...")
        try:
            cam.set_pixel_format("Mono8")
            print("✓ Pixel format set to Mono8 (method 1)")
        except Exception as e:
            print(f"⚠ Method 1 failed: {e}")
            try:
                cam.PixelFormat.set("Mono8")
                print("✓ Pixel format set to Mono8 (method 2)")
            except Exception as e:
                print(f"⚠ Could not set pixel format to Mono8: {e}")
                print("⚠ Continuing anyway - some cameras might not support this")

        print(f"Starting streaming with buffer_count={self.buffer_count}...")
        cam.start_streaming(handler=self._on_frame, buffer_count=self.buffer_count)
        self._running = True
        print("✓ Camera streaming started successfully")
        print("=== VIMBA CAMERA STARTUP COMPLETE ===")

    def stop(self):
        if self._cam and self._running:
            self._cam.stop_streaming()
            # Exit camera context (closes)
            self._cam.__exit__(None, None, None)
            self._running = False

        if self._vimba_ctx:
            self._vimba_ctx.__exit__(None, None, None)
            self._vimba_ctx = None

    def set_exposure(self, exposure_us: float) -> None:
        """Set camera exposure time in microseconds."""
        if hasattr(self, '_cam') and self._cam:
            try:
                self._cam.ExposureTime.set(exposure_us)
                print(f"Successfully set exposure to {exposure_us} µs")
            except Exception as e:
                print(f"Failed to set exposure: {e}")
                # Try alternative method
                try:
                    self._cam.get_feature_by_name('ExposureTime').set(exposure_us)
                    print(f"Successfully set exposure to {exposure_us} µs (alternative method)")
                except Exception as e2:
                    print(f"Failed to set exposure with alternative method: {e2}")

    def set_gain(self, gain_db: float) -> None:
        """Set camera gain in dB."""
        if hasattr(self, '_cam') and self._cam:
            try:
                self._cam.Gain.set(gain_db)
                print(f"Successfully set gain to {gain_db} dB")
            except Exception as e:
                print(f"Failed to set gain: {e}")
                # Try alternative method
                try:
                    self._cam.get_feature_by_name('Gain').set(gain_db)
                    print(f"Successfully set gain to {gain_db} dB (alternative method)")
                except Exception as e2:
                    print(f"Failed to set gain with alternative method: {e2}")

    def set_frame_rate(self, fps: float):
        if self._cam:
            self._cam.AcquisitionFrameRate.set(fps)


class CameraController(BaseCamera):
    """Unified facade choosing real or mock camera automatically."""

    def __init__(self, frame_queue: Queue):
        self.q = frame_queue
        if not VIMBA_AVAILABLE:
            raise RuntimeError("VimbaPython SDK not found – install Allied Vision Vimba and ensure Python bindings are available.")

        # Instantiate real camera; raise any errors to caller instead of silently falling back.
        self.cam: BaseCamera = VimbaCamera(frame_queue)

    # Proxy methods ------------------------------------------------------
    def start(self):
        # Attempt to start the real Allied Vision camera. If this fails, we
        # deliberately propagate the exception instead of silently falling back
        # to a synthetic MockCamera so that issues are detected immediately.
        self.cam.start()

    def stop(self):
        self.cam.stop()

    def set_exposure(self, us):
        self.cam.set_exposure(us)

    def set_gain(self, db):
        self.cam.set_gain(db)

    def set_frame_rate(self, fps):
        self.cam.set_frame_rate(fps) 