#!/usr/bin/env python3
"""Test script for the new camera API endpoints."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test the new camera API endpoints."""
    
    print("🧪 Testing new camera API endpoints...")
    
    # Test 1: Get camera status
    print("\n1. Testing GET /api/camera/status")
    try:
        response = requests.get(f"{BASE_URL}/api/camera/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status retrieved: exposure={data.get('exposure')}µs, gain={data.get('gain')}dB")
        else:
            print(f"❌ Status request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status request error: {e}")
    
    # Test 2: Set exposure
    print("\n2. Testing POST /api/camera/exposure")
    try:
        test_exposure = 8000
        response = requests.post(
            f"{BASE_URL}/api/camera/exposure",
            json={"exposure": test_exposure}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Exposure set: {data.get('exposure')}µs")
        else:
            print(f"❌ Exposure request failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exposure request error: {e}")
    
    # Test 3: Set gain
    print("\n3. Testing POST /api/camera/gain")
    try:
        test_gain = 3.5
        response = requests.post(
            f"{BASE_URL}/api/camera/gain",
            json={"gain": test_gain}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Gain set: {data.get('gain')}dB")
        else:
            print(f"❌ Gain request failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Gain request error: {e}")
    
    # Test 4: Set both parameters at once
    print("\n4. Testing POST /api/camera/parameters")
    try:
        response = requests.post(
            f"{BASE_URL}/api/camera/parameters",
            json={"exposure": 6000, "gain": 2.0}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Parameters set: {data.get('parameters')}")
        else:
            print(f"❌ Parameters request failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Parameters request error: {e}")
    
    # Test 5: Test validation (invalid exposure)
    print("\n5. Testing validation (invalid exposure)")
    try:
        response = requests.post(
            f"{BASE_URL}/api/camera/exposure",
            json={"exposure": 200000}  # Too high
        )
        if response.status_code == 400:
            print(f"✅ Validation working: {response.json().get('detail')}")
        else:
            print(f"❌ Validation failed: expected 400, got {response.status_code}")
    except Exception as e:
        print(f"❌ Validation test error: {e}")
    
    # Test 6: Test validation (invalid gain)
    print("\n6. Testing validation (invalid gain)")
    try:
        response = requests.post(
            f"{BASE_URL}/api/camera/gain",
            json={"gain": 30.0}  # Too high
        )
        if response.status_code == 400:
            print(f"✅ Validation working: {response.json().get('detail')}")
        else:
            print(f"❌ Validation failed: expected 400, got {response.status_code}")
    except Exception as e:
        print(f"❌ Validation test error: {e}")
    
    # Test 7: Get final status
    print("\n7. Getting final status")
    try:
        response = requests.get(f"{BASE_URL}/api/camera/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Final status: exposure={data.get('exposure')}µs, gain={data.get('gain')}dB")
        else:
            print(f"❌ Final status request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Final status request error: {e}")

if __name__ == "__main__":
    test_api_endpoints()
