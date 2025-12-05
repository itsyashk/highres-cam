import time
import requests
import websocket
import threading
import json
import sys

def test_api_status():
    print("Testing API Status...")
    try:
        r = requests.get("http://localhost:8000/api/camera/status")
        r.raise_for_status()
        data = r.json()
        print(f"✓ Status OK. Camera count: {data.get('camera_count', 'UNKNOWN')}")
        return data.get('camera_count', 0)
    except Exception as e:
        print(f"❌ API Status Failed: {e}")
        return 0

def test_stream(cam_index):
    uri = f"ws://localhost:8000/ws/{cam_index}"
    print(f"Connecting to Stream {cam_index} ({uri})...")
    
    def on_message(ws, message):
        print(f"✓ [Cam {cam_index}] Received frame ({len(message)} bytes)")
        ws.close()

    def on_error(ws, error):
        print(f"❌ [Cam {cam_index}] Error: {error}")

    def on_open(ws):
        print(f"✓ [Cam {cam_index}] Connected")

    ws = websocket.WebSocketApp(uri,
                                on_message=on_message,
                                on_error=on_error,
                                on_open=on_open)
    ws.run_forever()

if __name__ == "__main__":
    count = test_api_status()
    if count == 0:
        print("❌ No cameras detected or server not running correctly.")
        sys.exit(1)
        
    print(f"Attempting to verify streams for {count} cameras...")
    threads = []
    for i in range(count):
        t = threading.Thread(target=test_stream, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join(timeout=5)
        
    print("✓ Test Complete")
