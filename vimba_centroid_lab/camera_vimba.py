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

    def __init__(self, frame_queue: Queue, camera_id: Optional[str] = None, buffer_count: int = 32):
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
        print(f"DEBUG: _on_frame callback called for {self.camera_id}")
        try:
            status = frame.get_status()
            if status == FrameStatus.Complete:
                img = frame.as_numpy_ndarray()
                print(f"📸 COMPLETE Frame received from {self.camera_id}: {img.shape}")
                
                if not self.q.full():
                    self.q.put(img.copy())
                    print(f"📦 Frame queued for {self.camera_id}")
                else:
                    print(f"⚠ Frame queue full for {self.camera_id}, dropping frame")
            else:

                # Log incomplete frames with camera ID (less spammy - only first few)
                if not hasattr(self, '_incomplete_count'):
                    self._incomplete_count = 0
                self._incomplete_count += 1
                if self._incomplete_count <= 5 or self._incomplete_count % 100 == 0:
                    print(f"⚠ [{self.camera_id}] Incomplete frame #{self._incomplete_count}: status={status}")


            # CRITICAL: Always queue the frame back to the camera to keep streaming alive
            cam.queue_frame(frame)
            
        except Exception as e:
            print(f"❌ Frame handler error: {e}")
            import traceback
            traceback.print_exc()
            # Even on error, try to recycle the frame
            try:
                cam.queue_frame(frame)
            except:
                print(f"❌ Failed to recycle frame after error")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        print("=== VIMBA CAMERA STARTUP ===")
        
        # NOTE: self._vimba_ctx must be managed by the Controller to share Vimba instance across cameras
        # For this single camera class, we assume we are given a camera object from the controller
        
        if not self._cam:
             raise RuntimeError("Camera not initialized. Controller must assign a camera instance.")

        print(f"Opening camera {self.camera_id} (AccessMode.Full)...")
        from vimba import AccessMode  # type: ignore
        try:
            # Set access mode before entering context (as per VimbaPython API)
            self._cam.set_access_mode(AccessMode.Full)
            
            # Use context manager protocol which is standard for VimbaPython
            # We manually call __enter__ to keep it open until stop() is called
            self._cam.__enter__()
            print(f"✓ Camera {self.camera_id} opened successfully")
        except Exception as e:
            print(f"❌ Failed to open camera {self.camera_id}: {e}")
            raise

        # Adjust packet size for GigE, ignore errors
        print("Attempting to adjust GigE packet size...")
        try:
            self._cam.GVSPAdjustPacketSize.run()
            while not self._cam.GVSPAdjustPacketSize.is_done():
                pass
            print("✓ GigE packet size adjusted")
        except Exception as e:
            print(f"⚠ GigE packet size adjustment failed (not a GigE camera?): {e}")

        # Set Bandwidth Limit for multi-camera operation
        # 31 MB/s is camera minimum - trying this for dual-camera stability
        TARGET_BANDWIDTH = 31_000_000
        print(f"Setting StreamBytesPerSecond to {TARGET_BANDWIDTH}...")



        try:
            # Try setting StreamBytesPerSecond directly
            try:
                feat = self._cam.get_feature_by_name("StreamBytesPerSecond")
                feat.set(TARGET_BANDWIDTH)
                print(f"✓ StreamBytesPerSecond set to {TARGET_BANDWIDTH}")
            except Exception:
                # Fallback to DeviceLinkThroughputLimit
                feat = self._cam.get_feature_by_name("DeviceLinkThroughputLimit")
                feat.set(TARGET_BANDWIDTH)
                print(f"✓ DeviceLinkThroughputLimit set to {TARGET_BANDWIDTH}")
        except Exception as e:
            print(f"⚠ Failed to set bandwidth limit: {e}")

        # NEW: Set Inter-Packet Delay for multi-camera GigE stability
        # This gives the NIC time to process each packet, reducing frame loss
        print("Setting GigE inter-packet delay (GevSCPD)...")
        try:
            # GevSCPD = Stream Channel Packet Delay (in clock ticks)
            # Higher values = slower streaming but more reliable
            feat = self._cam.get_feature_by_name("GevSCPD")
            feat.set(10000)  # ~10000 ticks delay
            print(f"✓ GevSCPD set to 10000")
        except Exception as e:
            print(f"⚠ Failed to set GevSCPD: {e}")
            
        # Also try frame transmission delay if available
        try:
            feat = self._cam.get_feature_by_name("GevSCFTD")
            feat.set(100000)  # Frame transmission delay
            print(f"✓ GevSCFTD set to 100000")
        except Exception as e:
            print(f"⚠ GevSCFTD not available: {e}")



        # Ensure Mono8 pixel format if supported by SDK/camera
        print("Setting pixel format to Mono8 if supported...")
        try:
            # Prefer enum-based API when available
            from vimba import PixelFormat as PF  # type: ignore
            try:
                self._cam.set_pixel_format(PF.Mono8)  # enum expected by some SDK versions
                print("✓ Pixel format set to Mono8 via set_pixel_format(enum)")
            except Exception as e1:
                print(f"⚠ set_pixel_format(enum) failed: {e1}")
                try:
                    feat = self._cam.get_feature_by_name('PixelFormat')
                    # Try enum value if present in feature API
                    try:
                        feat.set(PF.Mono8)  # some SDKs accept enum here
                        print("✓ Pixel format set to Mono8 via feature.set(enum)")
                    except Exception as e1b:
                        # Fallback to string name
                        feat.set('Mono8')
                        print("✓ Pixel format set to Mono8 via feature.set('Mono8')")
                except Exception as e2:
                    print(f"⚠ Feature API fallback failed: {e2}")
        except Exception as e:
            # If PF import or all attempts fail, just continue with current format
            print(f"⚠ Could not enforce Mono8 pixel format: {e}")
            print("⚠ Continuing with camera default pixel format")

        print(f"Starting streaming with buffer_count={self.buffer_count}...")
        self._cam.start_streaming(handler=self._on_frame, buffer_count=self.buffer_count)
        self._running = True
        print(f"✓ Camera {self.camera_id} streaming started successfully")

    def attach_camera(self, cam_obj):
        """Attaches a Vimba Camera object found by the controller."""
        self._cam = cam_obj
        self.camera_id = cam_obj.get_id()

    def stop(self):
        if self._cam and self._running:
            try:
                self._cam.stop_streaming()
            except Exception: 
                pass
            # Exit camera context (closes)
            self._cam.__exit__(None, None, None)
            self._running = False

    # Detached context management from VimbaCamera to CameraController

    def set_exposure(self, exposure_us: float) -> None:
        """Set camera exposure time in microseconds."""
        if hasattr(self, '_cam') and self._cam:
            try:
                self._cam.ExposureTime.set(exposure_us)
                # Verify the setting was applied
                actual_exposure = self._cam.ExposureTime.get()
                print(f"✓ Set exposure to {exposure_us} µs (actual: {actual_exposure} µs)")
            except Exception as e:
                print(f"❌ Failed to set exposure: {e}")
                # Try alternative method
                try:
                    feat = self._cam.get_feature_by_name('ExposureTime')
                    feat.set(exposure_us)
                    actual_exposure = feat.get()
                    print(f"✓ Set exposure to {exposure_us} µs via feature (actual: {actual_exposure} µs)")
                except Exception as e2:
                    print(f"❌ Failed to set exposure with alternative method: {e2}")
        else:
            print(f"❌ Camera not available for exposure setting")

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
        print("=== CAMERA CONTROLLER STARTUP ===")
        
        # 1. Try Vimba
        if self.Vimba:
            try:
                self._vimba_ctx = self.Vimba.get_instance()
                self._vimba_ctx.__enter__()
                
                cams_found = self._vimba_ctx.get_all_cameras()
                print(f"✓ Vimba found {len(cams_found)} camera(s)")
                
                for cam_obj in cams_found:
                    q = Queue(maxsize=30)
                    vcam = VimbaCamera(q)
                    vcam.attach_camera(cam_obj)
                    self.cameras.append(vcam)
                    self.queues.append(q)

                    
            except Exception as e:
                print(f"❌ Error initializing Vimba: {e}")
                print(f"❌ Failed to initialize Vimba: {e}")
                if self._vimba_ctx:
                    self._vimba_ctx.__exit__(None, None, None)
                    self._vimba_ctx = None
        
        # 2. If no cameras found, do NOT fall back to mock cameras (per user request)
        if len(self.cameras) == 0:
            print("❌ No cameras detected! Please check connections.")
                
        # 3. Start all cameras with staggered delays
        import time
        for i, cam in enumerate(self.cameras):
            print(f"Starting camera #{i}...")
            cam.start()
            if i < len(self.cameras) - 1:
                print(f"⏳ Waiting 3 seconds before starting next camera...")
                time.sleep(3)  # Stagger to prevent simultaneous stream bursts
            
        print(f"✓ All {len(self.cameras)} cameras started.")


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
 