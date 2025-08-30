# New Frontend-Backend Communication Protocol

## Overview

This document describes the improved frontend-backend communication protocol for the Vimba Centroid Lab camera control system. The new implementation replaces the "cooked" communication with a robust, real-time system.

## Key Improvements

### 1. **Dual WebSocket Architecture**
- **State WebSocket** (`/ws/camera`): Real-time camera parameter updates
- **Video WebSocket** (`/ws`): Live video stream

### 2. **Improved REST API**
- **Validation**: Input validation for exposure (100-100000µs) and gain (0-24dB)
- **Error Handling**: Comprehensive error responses with meaningful messages
- **Batch Updates**: Set multiple parameters at once
- **Status Endpoint**: Get comprehensive camera status

### 3. **Frontend Enhancements**
- **Debounced Sliders**: Prevents spam requests (300ms delay)
- **Real-time Updates**: UI updates automatically via WebSocket
- **Error Recovery**: UI reverts to previous values on errors
- **Notifications**: User-friendly error/success messages
- **Connection Resilience**: Automatic WebSocket reconnection

## API Endpoints

### Camera Status
```http
GET /api/camera/status
```
Returns current camera parameters and state.

### Set Exposure
```http
POST /api/camera/exposure
Content-Type: application/json

{
  "exposure": 5000
}
```
Sets camera exposure time in microseconds (100-100000µs).

### Set Gain
```http
POST /api/camera/gain
Content-Type: application/json

{
  "gain": 2.5
}
```
Sets camera gain in decibels (0-24dB).

### Set Multiple Parameters
```http
POST /api/camera/parameters
Content-Type: application/json

{
  "exposure": 6000,
  "gain": 3.0
}
```
Sets multiple camera parameters at once.

## WebSocket Messages

### State WebSocket (`/ws/camera`)

#### Incoming Messages
```json
{"type": "ping"}
{"type": "get_status"}
```

#### Outgoing Messages
```json
{"type": "initial_state", "data": {...}}
{"type": "exposure_updated", "exposure": 5000, "timestamp": 1234567890}
{"type": "gain_updated", "gain": 2.5, "timestamp": 1234567890}
{"type": "parameters_updated", "parameters": {...}, "timestamp": 1234567890}
{"type": "status_update", "data": {...}}
{"type": "pong"}
```

## Frontend Features

### Debounced Sliders
- Sliders wait 300ms after user stops moving before sending request
- Prevents excessive API calls during rapid slider movement
- Immediate UI feedback with delayed backend update

### Real-time Synchronization
- WebSocket broadcasts parameter changes to all connected clients
- UI automatically updates when parameters change from other sources
- Maintains consistency across multiple browser tabs/windows

### Error Handling
- Invalid values are rejected with descriptive error messages
- UI reverts to previous valid values on errors
- User notifications for success/error states
- Automatic retry of failed requests

### Connection Management
- Automatic WebSocket reconnection on disconnection
- Periodic ping/pong to keep connections alive
- Graceful degradation when WebSocket is unavailable

## Usage Examples

### Basic Usage
1. Start the server: `python -m uvicorn vimba_centroid_lab.web_backend:app --host 0.0.0.0 --port 8000`
2. Open browser to `http://localhost:8000`
3. Use sliders to adjust exposure and gain
4. Watch real-time updates and notifications

### Programmatic Usage
```python
import requests

# Get current status
status = requests.get("http://localhost:8000/api/camera/status").json()

# Set exposure
response = requests.post("http://localhost:8000/api/camera/exposure", 
                        json={"exposure": 8000})

# Set both parameters
response = requests.post("http://localhost:8000/api/camera/parameters",
                        json={"exposure": 6000, "gain": 2.5})
```

### Testing
Run the test script to verify API functionality:
```bash
python test_new_api.py
```

## Benefits

1. **Reliability**: Robust error handling and validation
2. **Performance**: Debounced requests prevent spam
3. **User Experience**: Real-time updates and notifications
4. **Scalability**: WebSocket broadcasting supports multiple clients
5. **Maintainability**: Clean separation of concerns
6. **Debugging**: Comprehensive logging and error messages

## Migration from Old System

The new system is backward compatible with the existing camera controller interface. The main changes are:

- New API endpoints with validation
- WebSocket-based real-time updates
- Improved frontend with debouncing and error handling
- Better error messages and user feedback

The core camera control logic remains unchanged, ensuring compatibility with existing Vimba camera hardware.

