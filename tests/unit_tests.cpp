/**
 * Unit tests for Monica Image Processing HTTP Server
 * These tests verify the core functionality of the server components
 */

#include <iostream>
#include <cassert>
#include <string>
#include "../include/server/LoggingUtils.h"

// Test function declarations
void test_logging_utils();

int main() {
    std::cout << "Running unit tests for Monica Image Processing HTTP Server...\n";
    std::cout << "================================================================\n";

    try {
        test_logging_utils();
        
        std::cout << "\n🎉 All unit tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Unit test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

void test_logging_utils() {
    std::cout << "Testing LoggingUtils functions...\n";
    
    // Test request ID generation
    std::string requestId1 = RequestLogger::generateRequestId();
    std::string requestId2 = RequestLogger::generateRequestId();
    
    // Verify that request IDs are not empty and are different
    assert(!requestId1.empty());
    assert(!requestId2.empty());
    assert(requestId1 != requestId2);
    
    // Test starting a request
    auto metrics = RequestLogger::startRequest("/test");
    assert(metrics != nullptr);
    assert(!metrics->requestId.empty());
    assert(metrics->endpoint == "/test");
    
    std::cout << "✓ LoggingUtils functions work correctly\n";
}