#!/usr/bin/env python3
"""
Test script for Monica Image Processing HTTP Server
Tests all available endpoints with sample requests
"""

import requests
import json
import os
import sys
from io import BytesIO

# Server configuration
SERVER_URL = "http://localhost:8080"
TEST_IMAGE_PATH = "test_image.jpg"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200 and response.text == "OK":
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Health check failed with exception: {e}")
        return False

def test_version_endpoint():
    """Test the version endpoint"""
    print("\nTesting /version endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/version")
        if response.status_code == 200:
            version_info = response.json()
            print(f"✓ Version check passed: {version_info}")
            return True
        else:
            print(f"✗ Version check failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Version check failed with exception: {e}")
        return False

def test_sketch_drawing():
    """Test the sketch drawing endpoint"""
    print("\nTesting /api/sketchDrawing endpoint...")
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"⚠ Skipping sketch drawing test - {TEST_IMAGE_PATH} not found")
            return True
            
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{SERVER_URL}/api/sketchDrawing", files=files)
            
        if response.status_code == 200:
            print("✓ Sketch drawing test passed")
            return True
        else:
            print(f"✗ Sketch drawing test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Sketch drawing test failed with exception: {e}")
        return False

def test_face_detect():
    """Test the face detection endpoint"""
    print("\nTesting /api/faceDetect endpoint...")
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"⚠ Skipping face detect test - {TEST_IMAGE_PATH} not found")
            return True
            
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{SERVER_URL}/api/faceDetect", files=files)
            
        if response.status_code == 200:
            print("✓ Face detect test passed")
            return True
        else:
            print(f"✗ Face detect test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Face detect test failed with exception: {e}")
        return False

def test_face_landmark():
    """Test the face landmark endpoint"""
    print("\nTesting /api/faceLandMark endpoint...")
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"⚠ Skipping face landmark test - {TEST_IMAGE_PATH} not found")
            return True
            
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{SERVER_URL}/api/faceLandMark", files=files)
            
        if response.status_code == 200:
            print("✓ Face landmark test passed")
            return True
        else:
            print(f"✗ Face landmark test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Face landmark test failed with exception: {e}")
        return False

def test_cartoon():
    """Test the cartoon endpoint"""
    print("\nTesting /api/cartoon endpoint...")
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"⚠ Skipping cartoon test - {TEST_IMAGE_PATH} not found")
            return True
            
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            # Test with type parameter
            response = requests.post(f"{SERVER_URL}/api/cartoon?type=1", files=files)
            
        if response.status_code == 200:
            print("✓ Cartoon test passed")
            return True
        else:
            print(f"✗ Cartoon test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Cartoon test failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print(f"Running tests against server at {SERVER_URL}")
    print("=" * 50)
    
    # Check if server is running
    try:
        requests.get(SERVER_URL, timeout=5)
    except:
        print("⚠ Server doesn't seem to be running. Please start the server first:")
        print("   cd /Users/tony/CLionProjects/MonicaImageProcessHttpServer/src/build")
        print("   ./MonicaImageProcessHttpServer")
        return
    
    # Run all tests
    tests = [
        test_health_endpoint,
        test_version_endpoint,
        test_sketch_drawing,
        test_face_detect,
        test_face_landmark,
        test_cartoon
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {total - passed} tests failed")

if __name__ == "__main__":
    main()