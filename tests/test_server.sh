#!/bin/bash

# Test script for Monica Image Processing HTTP Server
# Tests all available endpoints with curl

SERVER_URL="http://localhost:8080"
TEST_IMAGE="test_image.jpg"

echo "Running tests against server at $SERVER_URL"
echo "=================================================="

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local description=$3
    local data=${4:-}
    
    echo "Testing $description..."
    
    if [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt -X POST "$SERVER_URL$endpoint" -d "$data")
        elif [ -f "$TEST_IMAGE" ]; then
            response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt -X POST "$SERVER_URL$endpoint" --data-binary "@$TEST_IMAGE")
        else
            echo "⚠ Skipping $description - $TEST_IMAGE not found"
            return 0
        fi
    else
        response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt "$SERVER_URL$endpoint")
    fi
    
    http_code=${response: -3}
    
    if [ "$http_code" = "200" ]; then
        echo "✓ $description passed"
        return 0
    else
        echo "✗ $description failed: HTTP $http_code"
        echo "Response: $(cat /tmp/curl_response.txt)"
        return 1
    fi
}

# Check if server is running
echo "Checking if server is running..."
curl -s --connect-timeout 5 "$SERVER_URL/health" > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠ Server doesn't seem to be running. Please start the server first:"
    echo "   cd /Users/tony/CLionProjects/MonicaImageProcessHttpServer/src/build"
    echo "   ./MonicaImageProcessHttpServer"
    exit 1
fi

# Run tests
passed=0
total=0

# Health endpoint
total=$((total + 1))
if test_endpoint "/health" "GET" "Health check"; then
    passed=$((passed + 1))
fi

# Version endpoint
total=$((total + 1))
if test_endpoint "/version" "GET" "Version check"; then
    passed=$((passed + 1))
fi

# Sketch drawing endpoint
total=$((total + 1))
if test_endpoint "/api/sketchDrawing" "POST" "Sketch drawing"; then
    passed=$((passed + 1))
fi

# Face detect endpoint
total=$((total + 1))
if test_endpoint "/api/faceDetect" "POST" "Face detection"; then
    passed=$((passed + 1))
fi

# Face landmark endpoint
total=$((total + 1))
if test_endpoint "/api/faceLandMark" "POST" "Face landmarks"; then
    passed=$((passed + 1))
fi

# Cartoon endpoint
total=$((total + 1))
if test_endpoint "/api/cartoon?type=1" "POST" "Cartoon effect"; then
    passed=$((passed + 1))
fi

# Summary
echo ""
echo "=================================================="
echo "Test Results: $passed/$total tests passed"

if [ $passed -eq $total ]; then
    echo "🎉 All tests passed!"
else
    echo "❌ $((total - passed)) tests failed"
fi

# Clean up
rm -f /tmp/curl_response.txt