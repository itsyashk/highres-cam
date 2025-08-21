#!/usr/bin/env python3
"""Simple camera test script."""

try:
    from vimba import Vimba, Camera
    print("✅ Vimba imported successfully")
    
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        print(f"✅ Found {len(cams)} camera(s)")
        
        if cams:
            with cams[0] as cam:
                print(f"✅ Camera opened successfully")
                print(f"   Camera ID: {cam.get_id()}")
                
                # Try to set pixel format
                try:
                    cam.set_pixel_format("Mono8")
                    print("✅ Pixel format set to Mono8")
                except Exception as e:
                    print(f"⚠️  Failed to set pixel format: {e}")
                
                # Try to get a frame
                try:
                    frame = cam.get_frame()
                    print(f"✅ Got frame: {frame.get_width()}x{frame.get_height()}")
                except Exception as e:
                    print(f"⚠️  Failed to get frame: {e}")
                    
except Exception as e:
    print(f"❌ Error: {e}")
