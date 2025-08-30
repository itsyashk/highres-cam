"""FastAPI backend serving camera frames over WebSocket."""
from __future__ import annotations

import asyncio
import cv2
import numpy as np
import json
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional
import csv
import os
from pathlib import Path  # NEW: for saving photos

from vimba import Vimba, FrameStatus, Camera, PixelFormat
from .camera_vimba import CameraController
from .processing import detect_blobs, baseline_centroid, subpixel_centroid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Separate queues for camera output and WebSocket streaming
camera_queue: Queue[np.ndarray] = Queue(maxsize=10)
websocket_queue: Queue[np.ndarray] = Queue(maxsize=10)
cam = CameraController(camera_queue)
photo_save_queue: "queue.Queue[np.ndarray]" | None = None  # NEW global queue for async saving

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Global state with validation
camera_state = {
    "exposure": 5000,
    "gain": 0,
    "streaming": False,
    "capture_series": False,
    "series_data": [],
    "series_target": 50,
    "pixel_size_mm": 0.6,
    "known_diameter_mm": 12.7,
    "selected_blob": None,
    "baseline_centroid": None,
    "subpixel_centroid": None,
    "current_frame": None,
    "camera_controller": cam,
    "last_update": time.time(),
    "save_photos": False,          # NEW: flag to trigger photo burst
    "save_remaining": 0,           # NEW: remaining photos to save
    "save_target": 1000            # NEW: desired photo count
}

# Validation functions
def validate_exposure(exposure: float) -> bool:
    """Validate exposure value is within reasonable range."""
    return 100 <= exposure <= 100000  # 100µs to 100ms

def validate_gain(gain: float) -> bool:
    """Validate gain value is within reasonable range."""
    return 0 <= gain <= 24  # 0-24 dB typical range

# Improved API endpoints
@app.get("/api/camera/status")
async def get_camera_status():
    """Get comprehensive camera status including current parameters."""
    return {
        "exposure": camera_state["exposure"],
        "gain": camera_state["gain"],
        "streaming": camera_state["streaming"],
        "capture_series": camera_state["capture_series"],
        "series_progress": len(camera_state["series_data"]),
        "series_target": camera_state["series_target"],
        "pixel_size_mm": camera_state["pixel_size_mm"],
        "known_diameter_mm": camera_state["known_diameter_mm"],
        "selected_blob": camera_state["selected_blob"],
        "last_update": camera_state["last_update"]
    }

@app.post("/api/camera/exposure")
async def set_exposure(data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Set camera exposure with validation and real-time updates."""
    try:
        exposure = float(data.get("exposure", 5_000_000))
        
        # Validate input
        if not validate_exposure(exposure):
            raise HTTPException(
                status_code=400, 
                detail=f"Exposure value {exposure}µs is outside valid range (100-100000µs)"
            )
        
        # Update camera
        if camera_state["camera_controller"]:
            camera_state["camera_controller"].set_exposure(exposure)
            camera_state["exposure"] = exposure
            camera_state["last_update"] = time.time()
            
            # Broadcast update to all connected clients
            background_tasks.add_task(
                manager.broadcast, 
                json.dumps({
                    "type": "exposure_updated",
                    "exposure": exposure,
                    "timestamp": camera_state["last_update"]
                })
            )
            
            return {
                "success": True, 
                "exposure": exposure,
                "message": f"Exposure set to {exposure}µs"
            }
        else:
            raise HTTPException(status_code=503, detail="Camera controller not available")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid exposure value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set exposure: {str(e)}")

@app.post("/api/camera/gain")
async def set_gain(data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Set camera gain with validation and real-time updates."""
    try:
        gain = float(data.get("gain", 0))
        
        # Validate input
        if not validate_gain(gain):
            raise HTTPException(
                status_code=400, 
                detail=f"Gain value {gain}dB is outside valid range (0-24dB)"
            )
        
        # Update camera
        if camera_state["camera_controller"]:
            camera_state["camera_controller"].set_gain(gain)
            camera_state["gain"] = gain
            camera_state["last_update"] = time.time()
            
            # Broadcast update to all connected clients
            background_tasks.add_task(
                manager.broadcast, 
                json.dumps({
                    "type": "gain_updated",
                    "gain": gain,
                    "timestamp": camera_state["last_update"]
                })
            )
            
            return {
                "success": True, 
                "gain": gain,
                "message": f"Gain set to {gain}dB"
            }
        else:
            raise HTTPException(status_code=503, detail="Camera controller not available")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid gain value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set gain: {str(e)}")

@app.post("/api/camera/parameters")
async def set_camera_parameters(data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Set multiple camera parameters at once."""
    try:
        updates = {}
        exposure = data.get("exposure")
        gain = data.get("gain")
        
        if exposure is not None:
            exposure = float(exposure)
            if not validate_exposure(exposure):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Exposure value {exposure}µs is outside valid range"
                )
            updates["exposure"] = exposure
            
        if gain is not None:
            gain = float(gain)
            if not validate_gain(gain):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Gain value {gain}dB is outside valid range"
                )
            updates["gain"] = gain
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid parameters provided")
        
        # Apply updates
        if camera_state["camera_controller"]:
            for param, value in updates.items():
                if param == "exposure":
                    camera_state["camera_controller"].set_exposure(value)
                    camera_state["exposure"] = value
                elif param == "gain":
                    camera_state["camera_controller"].set_gain(value)
                    camera_state["gain"] = value
            
            camera_state["last_update"] = time.time()
            
            # Broadcast update
            background_tasks.add_task(
                manager.broadcast, 
                json.dumps({
                    "type": "parameters_updated",
                    "parameters": updates,
                    "timestamp": camera_state["last_update"]
                })
            )
            
            return {
                "success": True,
                "parameters": updates,
                "message": f"Updated parameters: {', '.join(f'{k}={v}' for k, v in updates.items())}"
            }
        else:
            raise HTTPException(status_code=503, detail="Camera controller not available")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set parameters: {str(e)}")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/camera")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "data": {
                "exposure": camera_state["exposure"],
                "gain": camera_state["gain"],
                "streaming": camera_state["streaming"]
            }
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "get_status":
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "data": await get_camera_status()
                }))
            elif message.get("type") == "set_exposure":
                exposure = float(message.get("exposure", 0))
                if validate_exposure(exposure):
                    if camera_state["camera_controller"]:
                        camera_state["camera_controller"].set_exposure(exposure)
                    camera_state["exposure"] = exposure
                    camera_state["last_update"] = time.time()
                    await manager.broadcast(json.dumps({
                        "type": "exposure_updated",
                        "exposure": exposure,
                        "timestamp": camera_state["last_update"]
                    }))
                else:
                    await websocket.send_text(json.dumps({"type":"error","message":"invalid_exposure"}))
            elif message.get("type") == "set_gain":
                gain = float(message.get("gain", 0))
                if validate_gain(gain):
                    if camera_state["camera_controller"]:
                        camera_state["camera_controller"].set_gain(gain)
                    camera_state["gain"] = gain
                    camera_state["last_update"] = time.time()
                    await manager.broadcast(json.dumps({
                        "type": "gain_updated",
                        "gain": gain,
                        "timestamp": camera_state["last_update"]
                    }))
                else:
                    await websocket.send_text(json.dumps({"type":"error","message":"invalid_gain"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# WebSocket endpoint for video streaming
@app.websocket("/ws")
async def video_websocket_endpoint(websocket: WebSocket):
    print("🔌 Video WebSocket connection established")
    await websocket.accept()
    frame_count = 0
    try:
        while True:
            # Wait for frames from the camera thread
            try:
                frame = websocket_queue.get(timeout=1.0)
                if frame is not None:
                    frame_count += 1
                    
                    # Convert to JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpeg_data = buffer.tobytes()
                    
                    # Send as binary WebSocket message
                    await websocket.send_bytes(jpeg_data)
                    print(f"✓ Frame #{frame_count} sent to browser")
            except Exception as e:
                if "Empty" in str(e):
                    # No frame available, send a keep-alive
                    print(f"⏳ No frame available, sending keepalive")
                    await websocket.send_text("keepalive")
                else:
                    print(f"❌ Video WebSocket error: {e}")
                    import traceback
                    traceback.print_exc()
                    break
    except WebSocketDisconnect:
        print("🔌 Video WebSocket disconnected")
    except Exception as e:
        print(f"❌ Video WebSocket exception: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/capture-series")
async def start_capture_series(data: Dict[str, Any]):
    camera_state["capture_series"] = True
    camera_state["series_target"] = data.get("frames", 50)
    camera_state["series_data"] = []
    return {"success": True, "target_frames": camera_state["series_target"]}

@app.post("/api/stop-series")
async def stop_capture_series():
    camera_state["capture_series"] = False
    if camera_state["series_data"]:
        # Save CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"output/series_{timestamp}.csv"
        os.makedirs("output", exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=camera_state["series_data"][0].keys())
            writer.writeheader()
            writer.writerows(camera_state["series_data"])
    
    return {"success": True, "frames_captured": len(camera_state["series_data"])}

@app.post("/api/calibrate")
async def calibrate_scale(data: Dict[str, Any]):
    if not camera_state["series_data"]:
        raise HTTPException(status_code=400, detail="No measurement data available")
    
    last_data = camera_state["series_data"][-1]
    measured_diameter_px = last_data.get("diameter_px", 0)
    known_mm = data.get("known_diameter_mm", 12.7)
    
    if measured_diameter_px > 0:
        camera_state["pixel_size_mm"] = known_mm / measured_diameter_px
        camera_state["known_diameter_mm"] = known_mm
    
    return {"success": True, "pixel_size_mm": camera_state["pixel_size_mm"]}

@app.post("/api/select-blob")
async def select_blob(data: Dict[str, Any]):
    x, y = data.get("x", 0), data.get("y", 0)
    print(f"🎯 BLOB SELECTED: x={x}, y={y}")
    camera_state["selected_blob"] = (x, y)
    return {"success": True, "selected": (x, y)}

@app.get("/api/zoom-view")
async def get_zoom_view():
    """Get a zoomed view around the selected blob."""
    if not camera_state["selected_blob"] or camera_state["current_frame"] is None:
        raise HTTPException(status_code=400, detail="No blob selected or no frame available")
    
    try:
        x, y = camera_state["selected_blob"]
        frame = camera_state["current_frame"]
        
        # Extract ROI around the selected point
        roi_size = 200  # pixels
        h, w = frame.shape[:2]
        
        # Calculate ROI bounds
        x1 = max(0, int(x - roi_size // 2))
        y1 = max(0, int(y - roi_size // 2))
        x2 = min(w, x1 + roi_size)
        y2 = min(h, y1 + roi_size)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpeg_data = buffer.tobytes()
        
        print(f"🔍 ZOOM VIEW: ROI extracted at ({x1},{y1}) to ({x2},{y2}), size: {roi.shape}")
        
        return HTMLResponse(content=jpeg_data, media_type="image/jpeg")
        
    except Exception as e:
        print(f"❌ ZOOM VIEW ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-photos")  # NEW ENDPOINT
async def save_photos(data: Dict[str, Any]):
    """Begin saving a burst of photos to tests/1000photos directory."""
    frames = int(data.get("frames", 1000))
    if frames <= 0:
        raise HTTPException(status_code=400, detail="frames must be >0")
    camera_state["save_photos"] = True
    camera_state["save_remaining"] = frames
    camera_state["save_target"] = frames
    return {"success": True, "frames": frames}


@app.on_event("startup")
async def startup_event():
    # Start camera in background thread
    def _run():
        """Background thread for camera streaming."""
        print("=== CAMERA STARTUP SEQUENCE BEGIN ===")
        # Wait a bit to ensure any previous processes are fully cleaned up
        print("Waiting 2 seconds for cleanup...")
        time.sleep(2)
        
        try:
            print("Starting camera controller...")
            camera_state["camera_controller"].start()
            print("✓ Camera controller started successfully")
            camera_state["streaming"] = True
            print("✓ Camera streaming state set to True")
            
            frame_count = 0
            last_frame_time = time.time()
            
            # Process frames from the camera queue
            print("=== ENTERING FRAME PROCESSING LOOP ===")
            while True:
                try:
                    # Get frame from camera controller
                    #print(f"Waiting for frame from camera_queue (attempt {frame_count + 1})...")
                    frame = camera_queue.get(timeout=1.0)
                    
                    if frame is not None:
                        frame_count += 1
                        current_time = time.time()
                        fps = 1.0 / (current_time - last_frame_time) if frame_count > 1 else 0
                        last_frame_time = current_time
                        
                        print(f"✓ Received frame #{frame_count} (shape: {frame.shape}, dtype: {frame.dtype}, FPS: {fps:.1f})")
                        
                        camera_state["current_frame"] = frame.copy()
                        # -------------------------------------------------
                        # NEW: Photo burst saving
                        if camera_state["save_photos"] and camera_state["save_remaining"] > 0:
                            try:
                                idx = camera_state["save_target"] - camera_state["save_remaining"]
                                if photo_save_queue and not photo_save_queue.full():
                                    photo_save_queue.put((idx, frame.copy()))
                                    camera_state["save_remaining"] -= 1
                                    if camera_state["save_remaining"] == 0:
                                        camera_state["save_photos"] = False
                                        print("✓ Photo burst capture queued for saving")
                                else:
                                    print("⚠ Save queue full, skipping frame")
                            except Exception as e:
                                print(f"❌ Failed to queue photo: {e}")
                        # -------------------------------------------------
                        # Process frame for centroid analysis if blob is selected
                        if camera_state["selected_blob"]:
                            print(f"Processing centroid analysis for blob at {camera_state['selected_blob']}")
                            try:
                                # Detect blobs
                                blobs = detect_blobs(frame)
                                if blobs:
                                    print(f"✓ Detected {len(blobs)} blobs")
                                    # Use first detected blob or closest to selection
                                    blob = blobs[0]
                                    
                                    # Calculate centroids
                                    baseline = baseline_centroid(frame, mode="core")
                                    refined, radius, edge_pts = subpixel_centroid(frame, baseline)
                                    
                                    camera_state["baseline_centroid"] = baseline
                                    camera_state["subpixel_centroid"] = refined
                                    print(f"✓ Centroid calculated: baseline={baseline}, refined={refined}, radius={radius}")
                                    
                                    # Capture series data
                                    if camera_state["capture_series"] and len(camera_state["series_data"]) < camera_state["series_target"]:
                                        delta_px = np.hypot(refined[0] - baseline[0], refined[1] - baseline[1])
                                        series_entry = {
                                            "timestamp": datetime.now().isoformat(),
                                            "frame": len(camera_state["series_data"]),
                                            "baseline_x": baseline[0],
                                            "baseline_y": baseline[1],
                                            "refined_x": refined[0],
                                            "refined_y": refined[1],
                                            "radius_px": radius,
                                            "diameter_px": 2 * radius,
                                            "delta_px": delta_px,
                                            "delta_um": delta_px * camera_state["pixel_size_mm"] * 1000,
                                            "exposure": camera_state["exposure"],
                                            "gain": camera_state["gain"]
                                        }
                                        camera_state["series_data"].append(series_entry)
                                        print(f"✓ Added series data entry #{len(camera_state['series_data'])}")
                                        
                                        # Auto-stop when target reached
                                        if len(camera_state["series_data"]) >= camera_state["series_target"]:
                                            camera_state["capture_series"] = False
                                            print("✓ Series capture completed automatically")
                                else:
                                    print("⚠ No blobs detected in frame")
                            except Exception as e:
                                print(f"❌ Centroid analysis error: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Put frame in WebSocket queue for streaming
                        if not websocket_queue.full():
                            websocket_queue.put(frame.copy())
                            print(f"✓ Frame #{frame_count} queued for WebSocket (websocket_queue size: {websocket_queue.qsize()})")
                        else:
                            print(f"⚠ WebSocket queue full, dropping frame #{frame_count}")
                            
                except Exception as e:
                    if "Empty" not in str(e):  # Queue.Empty is expected
                        print(f"❌ Frame processing error: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"⏳ No frame available (timeout), continuing...")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Camera error: {e}")
            import traceback
            traceback.print_exc()
            camera_state["streaming"] = False
            print("❌ Camera streaming state set to False")

    print("Starting camera thread...")
    Thread(target=_run, daemon=True).start()
    print("✓ Camera thread started")


# No explicit shutdown handler; daemon thread ends with process


async def frame_publisher():
    while True:
        try:
            # Wait for frame from camera
            frame = await asyncio.get_event_loop().run_in_executor(None, websocket_queue.get)
            # Encode to JPEG
            ret, jpg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            await manager.broadcast(jpg.tobytes())
        except Exception as e:
            # If no frame available, just continue
            await asyncio.sleep(0.1)


@app.get("/")
async def index():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>850nm Vision Lab</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                    overflow-x: hidden;
                }
                
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    color: white;
                }
                
                .header h1 {
                    font-size: 3rem;
                    font-weight: 300;
                    margin-bottom: 10px;
                    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    background: linear-gradient(45deg, #fff, #f0f0f0);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                
                .header p {
                    font-size: 1.1rem;
                    opacity: 0.9;
                    font-weight: 300;
                }
                
                .main-content {
                    display: grid;
                    grid-template-columns: 350px 1fr;
                    gap: 25px;
                    align-items: start;
                }
                
                .control-panel {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 25px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    height: fit-content;
                }
                
                .control-section {
                    margin-bottom: 25px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid rgba(102, 126, 234, 0.2);
                }
                
                .control-section:last-child {
                    border-bottom: none;
                    margin-bottom: 0;
                }
                
                .control-section h3 {
                    color: #667eea;
                    margin-bottom: 15px;
                    font-size: 1.1rem;
                    font-weight: 600;
                }
                
                .slider-group {
                    margin-bottom: 15px;
                }
                
                .slider-label {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                    font-size: 0.9rem;
                    color: #666;
                }
                
                .slider {
                    width: 100%;
                    height: 6px;
                    border-radius: 3px;
                    background: #e0e0e0;
                    outline: none;
                    -webkit-appearance: none;
                }
                
                .slider::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 18px;
                    height: 18px;
                    border-radius: 50%;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    cursor: pointer;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                }
                
                .slider::-moz-range-thumb {
                    width: 18px;
                    height: 18px;
                    border-radius: 50%;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    cursor: pointer;
                    border: none;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                }
                
                .btn {
                    width: 100%;
                    padding: 12px 16px;
                    border: none;
                    border-radius: 10px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin-bottom: 10px;
                }
                
                .btn-primary {
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }
                
                .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
                }
                
                .btn-danger {
                    background: linear-gradient(45deg, #ff6b6b, #ee5a52);
                    color: white;
                    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
                }
                
                .btn-danger:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
                }
                
                .btn-secondary {
                    background: linear-gradient(45deg, #4ecdc4, #44a08d);
                    color: white;
                    box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
                }
                
                .btn-secondary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(78, 205, 196, 0.6);
                }
                
                .btn:disabled {
                    background: #e0e0e0;
                    color: #999;
                    cursor: not-allowed;
                    transform: none;
                    box-shadow: none;
                }
                
                .input-group {
                    margin-bottom: 15px;
                }
                
                .input-group label {
                    display: block;
                    margin-bottom: 5px;
                    font-size: 0.9rem;
                    color: #666;
                }
                
                .input-group input {
                    width: 100%;
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    font-size: 0.9rem;
                }
                
                .status-card {
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 12px;
                    padding: 15px;
                    margin-top: 15px;
                    border-left: 4px solid #667eea;
                }
                
                .status-indicator {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-weight: 600;
                    font-size: 0.9rem;
                }
                
                .status-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }
                
                .status-dot.ready { background: #ffd700; }
                .status-dot.streaming { background: #4CAF50; animation: pulse 1s infinite; }
                .status-dot.error { background: #f44336; }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .stats {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin-top: 15px;
                }
                
                .stat-card {
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                    border-radius: 10px;
                    padding: 12px;
                    text-align: center;
                    border: 1px solid rgba(102, 126, 234, 0.2);
                }
                
                .stat-value {
                    font-size: 1.2rem;
                    font-weight: bold;
                    color: #667eea;
                }
                
                .stat-label {
                    font-size: 0.8rem;
                    color: #666;
                    margin-top: 3px;
                }
                
                .stream-section {
                    display: grid;
                    grid-template-columns: 1fr 300px;
                    gap: 20px;
                }
                
                .stream-container {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 25px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    text-align: center;
                    position: relative;
                }
                
                #stream {
                    max-width: 100%;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                    cursor: crosshair;
                }
                
                #stream:hover {
                    transform: scale(1.01);
                }
                
                .no-stream {
                    color: #666;
                    font-size: 1.1rem;
                    font-style: italic;
                }
                
                .zoom-pane {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 15px 30px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    text-align: center;
                }
                
                .zoom-pane h3 {
                    color: #667eea;
                    margin-bottom: 15px;
                    font-size: 1rem;
                }
                
                #zoomImage {
                    max-width: 100%;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                }
                
                .centroid-info {
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 12px;
                    padding: 15px;
                    margin-top: 15px;
                    border-left: 4px solid #4CAF50;
                }
                
                .centroid-info h4 {
                    color: #4CAF50;
                    margin-bottom: 10px;
                    font-size: 0.9rem;
                }
                
                .centroid-data {
                    font-size: 0.8rem;
                    color: #666;
                    line-height: 1.4;
                }
                
                .series-progress {
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 12px;
                    padding: 15px;
                    margin-top: 15px;
                    border-left: 4px solid #ff9800;
                }
                
                .progress-bar {
                    width: 100%;
                    height: 8px;
                    background: #e0e0e0;
                    border-radius: 4px;
                    overflow: hidden;
                    margin-top: 8px;
                }
                
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(45deg, #ff9800, #ff5722);
                    transition: width 0.3s ease;
                }
                
                @media (max-width: 1200px) {
                    .main-content {
                        grid-template-columns: 1fr;
                    }
                    .stream-section {
                        grid-template-columns: 1fr;
                    }
                    .header h1 {
                        font-size: 2.5rem;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>850nm @ 20.2MP Vision Lab</h1>
                    <p>High-Precision Camera Centroid Analysis</p>
                </div>
                
                <div class="main-content">
                    <div class="control-panel">
                        <div class="control-section">
                            <h3>🎮 Stream Controls</h3>
                            <button id="startBtn" class="btn btn-primary" onclick="toggleStream()">
                                <span id="startText">▶️ Start Stream</span>
                            </button>
                            <button id="captureBtn" class="btn btn-secondary" onclick="captureFrame()" disabled>
                                📸 Capture Frame
                            </button>
                            <button id="savePhotosBtn" class="btn btn-secondary" onclick="savePhotos()" disabled>
                                💾 Save 1000 Photos
                            </button>
                        </div>

                        <div class="control-section">
                            <h3>🔎 Zoom Scale</h3>
                            <div class="slider-group">
                                <div class="slider-label">
                                    <span>Zoom (x)</span>
                                    <span id="zoomValue">16</span>
                                </div>
                                <input type="range" id="zoomSlider" class="slider" min="2" max="32" value="16">
                            </div>
                        </div>
                        
                        <div class="control-section">
                            <h3>⚙️ Camera Settings</h3>
                            <div class="slider-group">
                                <div class="slider-label">
                                    <span>Exposure (µs)</span>
                                    <span id="exposureValue">5000</span>
                                </div>
                                <input type="range" id="exposureSlider" class="slider" min="100" max="20000" value="5000">
                            </div>
                            <div class="slider-group">
                                <div class="slider-label">
                                    <span>Gain (dB)</span>
                                    <span id="gainValue">0</span>
                                </div>
                                <input type="range" id="gainSlider" class="slider" min="0" max="24" value="0" step="0.1">
                            </div>
                        </div>
                        
                        <div class="control-section">
                            <h3>📊 Capture Series</h3>
                            <div class="input-group">
                                <label>Number of Frames:</label>
                                <input type="number" id="seriesFrames" value="50" min="1" max="1000">
                            </div>
                            <button id="seriesBtn" class="btn btn-secondary" onclick="toggleSeries()" disabled>
                                📊 Start Series
                            </button>
                            <div class="series-progress" id="seriesProgress" style="display: none;">
                                <div style="font-size: 0.9rem; color: #ff9800; margin-bottom: 5px;">
                                    Capturing: <span id="seriesCount">0</span>/<span id="seriesTarget">50</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="control-section">
                            <h3>📏 Calibration</h3>
                            <div class="input-group">
                                <label>Known Diameter (mm):</label>
                                <input type="number" id="knownDiameter" value="12.7" step="0.1" min="0.1">
                            </div>
                            <button class="btn btn-secondary" onclick="calibrate()">
                                🎯 Calibrate Scale
                            </button>
                            <div style="font-size: 0.8rem; color: #666; margin-top: 8px;">
                                Pixel Size: <span id="pixelSize">0.600</span> mm/px
                            </div>
                        </div>
                        
                        <div class="status-card">
                            <div class="status-indicator">
                                <div id="statusDot" class="status-dot ready"></div>
                                <span id="status">Ready to start streaming</span>
                            </div>
                        </div>
                        
                        <div class="stats">
                            <div class="stat-card">
                                <div id="fpsValue" class="stat-value">0</div>
                                <div class="stat-label">FPS</div>
                            </div>
                            <div class="stat-card">
                                <div id="frameCount" class="stat-value">0</div>
                                <div class="stat-label">Frames</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="stream-section">
                        <div class="stream-container">
                            <img id="stream" style="display: none;" alt="Live camera stream" onclick="handleImageClick(event)"/>
                            <div id="noStream" class="no-stream">
                                <div style="font-size: 3rem; margin-bottom: 15px;">📹</div>
                                Click "Start Stream" to begin live feed<br>
                                <small style="color: #999;">Click on the image to select a blob for analysis</small>
                            </div>
                        </div>
                        
                        <div class="zoom-pane">
                            <h3>🔍 Zoom View</h3>
                            <img id="zoomImage" style="display: none;" alt="Zoom view"/>
                            <div id="noZoom" style="color: #999; font-size: 0.9rem;">
                                Select a blob to see zoom view
                            </div>
                            
                            <div class="centroid-info" id="centroidInfo" style="display: none;">
                                <h4>🎯 Centroid Analysis</h4>
                                <div class="centroid-data" id="centroidData">
                                    Loading...
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // Global variables
                let ws = null;
                let stateWs = null;
                let isStreaming = false;
                let frameCount = 0;
                let lastFrameTime = 0;
                let fps = 0;
                let selectedBlob = null;
                
                // Debouncing variables
                let exposureTimeout = null;
                let gainTimeout = null;
                const DEBOUNCE_DELAY = 300; // ms
                
                // DOM elements
                const img = document.getElementById('stream');
                const noStream = document.getElementById('noStream');
                const startBtn = document.getElementById('startBtn');
                const startText = document.getElementById('startText');
                const captureBtn = document.getElementById('captureBtn');
                const savePhotosBtn = document.getElementById('savePhotosBtn'); // NEW
                const seriesBtn = document.getElementById('seriesBtn');
                const status = document.getElementById('status');
                const statusDot = document.getElementById('statusDot');
                const fpsValue = document.getElementById('fpsValue');
                const frameCountElement = document.getElementById('frameCount');
                const exposureSlider = document.getElementById('exposureSlider');
                const exposureValue = document.getElementById('exposureValue');
                const gainSlider = document.getElementById('gainSlider');
                const gainValue = document.getElementById('gainValue');
                const seriesProgress = document.getElementById('seriesProgress');
                const seriesCount = document.getElementById('seriesCount');
                const seriesTarget = document.getElementById('seriesTarget');
                const progressFill = document.getElementById('progressFill');
                const centroidInfo = document.getElementById('centroidInfo');
                const centroidData = document.getElementById('centroidData');
                const pixelSize = document.getElementById('pixelSize');
                const zoomSlider = document.getElementById('zoomSlider');
                const zoomValue = document.getElementById('zoomValue');

                // Initialize WebSocket connection for state updates
                function initStateWebSocket() {
                    if (stateWs) {
                        stateWs.close();
                    }
                    
                    stateWs = new WebSocket(`ws://${location.host}/ws/camera`);
                    
                    stateWs.onopen = () => {
                        console.log('🔌 State WebSocket connected');
                    };
                    
                    stateWs.onmessage = (event) => {
                        try {
                            const message = JSON.parse(event.data);
                            handleStateMessage(message);
                        } catch (e) {
                            console.error('Failed to parse state message:', e);
                        }
                    };
                    
                    stateWs.onclose = () => {
                        console.log('🔌 State WebSocket disconnected');
                        // Reconnect after 2 seconds
                        setTimeout(initStateWebSocket, 2000);
                    };
                    
                    stateWs.onerror = (error) => {
                        console.error('State WebSocket error:', error);
                    };
                }
                
                // Handle incoming state messages
                function handleStateMessage(message) {
                    switch (message.type) {
                        case 'initial_state':
                            updateUIFromState(message.data);
                            break;
                        case 'exposure_updated':
                            updateExposureUI(message.exposure);
                            break;
                        case 'gain_updated':
                            updateGainUI(message.gain);
                            break;
                        case 'parameters_updated':
                            updateUIFromParameters(message.parameters);
                            break;
                        case 'status_update':
                            updateUIFromState(message.data);
                            break;
                        case 'pong':
                            // Keep-alive response
                            break;
                        default:
                            console.log('Unknown message type:', message.type);
                    }
                }
                
                // Update UI from state data
                function updateUIFromState(state) {
                    if (state.exposure !== undefined) {
                        updateExposureUI(state.exposure);
                    }
                    if (state.gain !== undefined) {
                        updateGainUI(state.gain);
                    }
                    if (state.streaming !== undefined) {
                        // Handle streaming state if needed
                    }
                }
                
                // Update UI from parameters
                function updateUIFromParameters(parameters) {
                    if (parameters.exposure !== undefined) {
                        updateExposureUI(parameters.exposure);
                    }
                    if (parameters.gain !== undefined) {
                        updateGainUI(parameters.gain);
                    }
                }
                
                // Update exposure UI without triggering slider events
                function updateExposureUI(exposure) {
                    exposureSlider.removeEventListener('input', debouncedUpdateExposure);
                    exposureSlider.value = exposure;
                    exposureValue.textContent = exposure;
                    exposureSlider.addEventListener('input', debouncedUpdateExposure);
                }
                
                // Update gain UI without triggering slider events
                function updateGainUI(gain) {
                    gainSlider.removeEventListener('input', debouncedUpdateGain);
                    gainSlider.value = gain;
                    gainValue.textContent = gain;
                    gainSlider.addEventListener('input', debouncedUpdateGain);
                }

                // Debounced exposure update
                function debouncedUpdateExposure() {
                    const value = exposureSlider.value;
                    exposureValue.textContent = value;
                    
                    if (exposureTimeout) {
                        clearTimeout(exposureTimeout);
                    }
                    
                    exposureTimeout = setTimeout(() => {
                        updateExposure(value);
                    }, DEBOUNCE_DELAY);
                }
                
                // Debounced gain update
                function debouncedUpdateGain() {
                    const value = gainSlider.value;
                    gainValue.textContent = value;
                    
                    if (gainTimeout) {
                        clearTimeout(gainTimeout);
                    }
                    
                    gainTimeout = setTimeout(() => {
                        updateGain(value);
                    }, DEBOUNCE_DELAY);
                }

                // Initialize sliders with debouncing
                console.log('🔧 SETUP: Initializing sliders with debouncing');
                exposureSlider.addEventListener('input', debouncedUpdateExposure);
                gainSlider.addEventListener('input', debouncedUpdateGain);
                
                // Initialize state WebSocket
                initStateWebSocket();
                
                // Send periodic pings to keep WebSocket alive
                setInterval(() => {
                    if (stateWs && stateWs.readyState === WebSocket.OPEN) {
                        stateWs.send(JSON.stringify({type: 'ping'}));
                    }
                }, 30000); // Every 30 seconds

                function updateStats() {
                    const now = Date.now();
                    if (lastFrameTime > 0) {
                        fps = Math.round(1000 / (now - lastFrameTime));
                    }
                    lastFrameTime = now;
                    frameCount++;
                    
                    fpsValue.textContent = fps;
                    frameCountElement.textContent = frameCount;
                }

                function toggleStream() {
                    if (!isStreaming) {
                        startStream();
                    } else {
                        stopStream();
                    }
                }

                function startStream() {
                    status.textContent = 'Connecting...';
                    statusDot.className = 'status-dot ready';
                    
                    ws = new WebSocket(`ws://${location.host}/ws/camera`);
                    ws.binaryType = 'arraybuffer';
                    
                    ws.onopen = () => {
                        isStreaming = true;
                        startText.textContent = '⏹️ Stop Stream';
                        startBtn.className = 'btn btn-danger';
                        captureBtn.disabled = false;
                        seriesBtn.disabled = false;
                        savePhotosBtn.disabled = false; // NEW enable when streaming
                        status.textContent = 'Streaming live feed';
                        statusDot.className = 'status-dot streaming';
                        img.style.display = 'inline';
                        noStream.style.display = 'none';
                        
                        // Reset stats
                        frameCount = 0;
                        fps = 0;
                        fpsValue.textContent = '0';
                        frameCountElement.textContent = '0';
                        
                        // Start polling for state updates
                        startStatePolling();
                    };
                    
                    ws.onmessage = ev => {
                        const blob = new Blob([ev.data], {type: 'image/jpeg'});
                        img.src = URL.createObjectURL(blob);
                        updateStats();
                    };
                    
                    ws.onclose = () => {
                        stopStream();
                    };
                    
                    ws.onerror = () => {
                        status.textContent = 'Connection error';
                        statusDot.className = 'status-dot error';
                        stopStream();
                    };
                }

                function stopStream() {
                    if (ws) {
                        ws.close();
                        ws = null;
                    }
                    isStreaming = false;
                    startText.textContent = '▶️ Start Stream';
                    startBtn.className = 'btn btn-primary';
                    captureBtn.disabled = true;
                    seriesBtn.disabled = true;
                    savePhotosBtn.disabled = true; // NEW disable when not streaming
                    status.textContent = 'Ready to start streaming';
                    statusDot.className = 'status-dot ready';
                    img.style.display = 'none';
                    noStream.style.display = 'block';
                    
                    // Stop polling
                    if (statePollInterval) {
                        clearInterval(statePollInterval);
                        statePollInterval = null;
                    }
                }

                let statePollInterval = null;

                function startStatePolling() {
                    // Poll for state updates every 5 seconds
                    statePollInterval = setInterval(() => {
                        fetch('/api/camera/status')
                            .then(response => response.json())
                            .then(state => {
                                updateState(state);
                            })
                            .catch(e => {
                                console.log('State poll error:', e);
                            });
                    }, 5000);
                }

                function updateState(state) {
                    try {
                        // Update exposure and gain
                        if (state.exposure !== undefined) {
                            exposureSlider.value = state.exposure;
                            exposureValue.textContent = state.exposure;
                        }
                        if (state.gain !== undefined) {
                            gainSlider.value = state.gain;
                            gainValue.textContent = state.gain;
                        }
                        
                        // Update series progress
                        if (state.capture_series) {
                            seriesProgress.style.display = 'block';
                            seriesCount.textContent = state.series_progress || 0;
                            seriesTarget.textContent = state.series_target || 50;
                            const progress = ((state.series_progress || 0) / (state.series_target || 50)) * 100;
                            progressFill.style.width = progress + '%';
                        } else {
                            seriesProgress.style.display = 'none';
                        }
                        
                        // Update pixel size
                        if (state.pixel_size_mm !== undefined) {
                            pixelSize.textContent = state.pixel_size_mm.toFixed(3);
                        }
                        
                        // Update zoom view periodically if blob is selected
                        if (selectedBlob && isStreaming) {
                            updateZoomView();
                        }
                    } catch (e) {
                        console.log('State update error:', e);
                    }
                }

                function updateExposure(value) {
                    console.log(`🎛️ Frontend: Setting exposure to ${value} µs via WS`);
                    if (stateWs && stateWs.readyState === WebSocket.OPEN) {
                        stateWs.send(JSON.stringify({type: 'set_exposure', exposure: Number(value)}));
                    } else {
                        fetch('/api/camera/exposure', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ exposure: Number(value) })
                        });
                    }
                }

                function updateGain(value) {
                    console.log(`🎛️ Frontend: Setting gain to ${value} dB via WS`);
                    if (stateWs && stateWs.readyState === WebSocket.OPEN) {
                        stateWs.send(JSON.stringify({type: 'set_gain', gain: Number(value)}));
                    } else {
                        fetch('/api/camera/gain', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ gain: Number(value) })
                        });
                    }
                }
                
                // Notification system
                function showNotification(message, type = 'info') {
                    // Create notification element
                    const notification = document.createElement('div');
                    notification.className = `notification ${type}`;
                    notification.textContent = message;
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        padding: 12px 20px;
                        border-radius: 8px;
                        color: white;
                        font-weight: 500;
                        z-index: 1000;
                        max-width: 300px;
                        word-wrap: break-word;
                        animation: slideIn 0.3s ease-out;
                    `;
                    
                    // Set background color based on type
                    switch (type) {
                        case 'error':
                            notification.style.backgroundColor = '#f44336';
                            break;
                        case 'success':
                            notification.style.backgroundColor = '#4CAF50';
                            break;
                        case 'warning':
                            notification.style.backgroundColor = '#ff9800';
                            break;
                        default:
                            notification.style.backgroundColor = '#2196F3';
                    }
                    
                    // Add to page
                    document.body.appendChild(notification);
                    
                    // Remove after 5 seconds
                    setTimeout(() => {
                        notification.style.animation = 'slideOut 0.3s ease-in';
                        setTimeout(() => {
                            if (notification.parentNode) {
                                notification.parentNode.removeChild(notification);
                            }
                        }, 300);
                    }, 5000);
                }
                
                // Add CSS animations for notifications
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes slideIn {
                        from { transform: translateX(100%); opacity: 0; }
                        to { transform: translateX(0); opacity: 1; }
                    }
                    @keyframes slideOut {
                        from { transform: translateX(0); opacity: 1; }
                        to { transform: translateX(100%); opacity: 0; }
                    }
                `;
                document.head.appendChild(style);

                function toggleSeries() {
                    const frames = parseInt(document.getElementById('seriesFrames').value);
                    if (!isStreaming) return;
                    
                    fetch('/api/capture-series', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({frames: frames})
                    }).then(() => {
                        seriesBtn.textContent = '⏹️ Stop Series';
                        seriesBtn.className = 'btn btn-danger';
                    });
                }

                function calibrate() {
                    const knownDiameter = parseFloat(document.getElementById('knownDiameter').value);
                    fetch('/api/calibrate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({known_diameter_mm: knownDiameter})
                    });
                }

                function handleImageClick(event) {
                    const rect = img.getBoundingClientRect();
                    const x = event.clientX - rect.left;
                    const y = event.clientY - rect.top;
                    
                    // Convert to image coordinates
                    const scaleX = img.naturalWidth / rect.width;
                    const scaleY = img.naturalHeight / rect.height;
                    const imgX = x * scaleX;
                    const imgY = y * scaleY;
                    
                    console.log(`🎯 Frontend: Clicked at screen (${x}, ${y}), image (${imgX.toFixed(1)}, ${imgY.toFixed(1)})`);
                    
                    selectedBlob = {x: imgX, y: imgY};
                    
                    // Send to backend
                    fetch('/api/select-blob', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({x: imgX, y: imgY})
                    }).then(response => {
                        if (response.ok) {
                            console.log('🎯 Backend: Blob selection confirmed');
                            // Update zoom view
                            updateZoomView();
                        }
                    });
                }
                
                function updateZoomView() {
                    if (selectedBlob) {
                        const zoomImg = document.getElementById('zoomImage');
                        zoomImg.style.display = 'inline';
                        document.getElementById('noZoom').style.display = 'none';
                        
                        // Add timestamp to prevent caching
                        zoomImg.src = '/api/zoom-view?' + new Date().getTime();
                        console.log('🔍 Frontend: Requesting zoom view');
                    }
                }
                
                function testExposureGain() {
                    console.log('🧪 TEST: Testing exposure and gain updates');
                    showNotification('Testing camera parameters...', 'info');
                    
                    // Test setting both parameters at once
                    fetch('/api/camera/parameters', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            exposure: 10000,
                            gain: 5.5
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (data.success) {
                            console.log('✅ Test parameters updated:', data.parameters);
                            showNotification('Test parameters applied successfully!', 'success');
                        } else {
                            throw new Error(data.message || 'Failed to update test parameters');
                        }
                    })
                    .catch(error => {
                        console.error('❌ Test failed:', error);
                        showNotification(`Test failed: ${error.message}`, 'error');
                    });
                }
                
                // Function to update both parameters at once
                function updateCameraParameters(exposure, gain) {
                    const updates = {};
                    if (exposure !== undefined) updates.exposure = exposure;
                    if (gain !== undefined) updates.gain = gain;
                    
                    if (Object.keys(updates).length === 0) {
                        console.warn('No parameters to update');
                        return;
                    }
                    
                    fetch('/api/camera/parameters', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(updates)
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (data.success) {
                            console.log('✅ Parameters updated:', data.parameters);
                            showNotification('Camera parameters updated successfully!', 'success');
                        } else {
                            throw new Error(data.message || 'Failed to update parameters');
                        }
                    })
                    .catch(error => {
                        console.error('❌ Parameter update failed:', error);
                        showNotification(`Parameter update failed: ${error.message}`, 'error');
                    });
                }

                function captureFrame() {
                    if (img.src) {
                        const link = document.createElement('a');
                        link.download = 'capture_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.jpg';
                        link.href = img.src;
                        link.click();
                        
                        status.textContent = 'Frame captured!';
                        setTimeout(() => {
                            if (isStreaming) status.textContent = 'Streaming live feed';
                        }, 2000);
                    }
                }

                function savePhotos() { // NEW
                    savePhotosBtn.disabled = true;
                    fetch('/api/save-photos', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({frames: 1000})
                    })
                    .then(resp => resp.json())
                    .then(data => {
                        if (data.success) {
                            showNotification('Started saving 1000 photos', 'success');
                        } else {
                            throw new Error('Failed to start saving photos');
                        }
                    })
                    .catch(err => {
                        console.error(err);
                        showNotification('Error: ' + err.message, 'error');
                    })
                    .finally(() => {
                        savePhotosBtn.disabled = false;
                    });
                }
            </script>
        </body>
        </html>
        """
    )


# Launch background publisher task
@app.on_event("startup")
async def _start_publisher():
    asyncio.create_task(frame_publisher()) 