//
// Created by Tony on 2025/12/3.
// Enhanced logging utilities with request tracking and performance metrics
//

#ifndef MONICAIMAGEPROCESS_LOGGINGUTILS_H
#define MONICAIMAGEPROCESS_LOGGINGUTILS_H

#include <string>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "../utils/aixlog.hpp"

class RequestLogger {
public:
    struct RequestMetrics {
        std::string requestId;
        std::string endpoint;
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point endTime;
        size_t requestSize;
        size_t responseSize;
        int statusCode;
        bool completed;
        
        RequestMetrics(const std::string& id, const std::string& ep) 
            : requestId(id), endpoint(ep), startTime(std::chrono::high_resolution_clock::now()),
              requestSize(0), responseSize(0), statusCode(0), completed(false) {}
    };

private:
    static std::atomic<uint64_t> requestCounter;
    static std::unordered_map<std::string, std::shared_ptr<RequestMetrics>> activeRequests;
    static std::mutex requestsMutex;
    static boost::uuids::random_generator uuidGenerator;

public:
    // Generate a unique request ID
    static std::string generateRequestId();
    
    // Start tracking a request
    static std::shared_ptr<RequestMetrics> startRequest(const std::string& endpoint);
    
    // End tracking a request
    static void endRequest(const std::shared_ptr<RequestMetrics>& metrics, int statusCode, size_t responseSize);
    
    // Log a message with request context
    static void logWithContext(AixLog::Severity severity, const std::string& message, 
                              const std::shared_ptr<RequestMetrics>& metrics = nullptr);
    
    // Get performance statistics
    static std::string getPerformanceStats();
    
    // Log performance metrics for a completed request
    static void logRequestPerformance(const std::shared_ptr<RequestMetrics>& metrics);
};

// Macro for easy logging with request context
#define LOG_REQ(severity, metrics, message) \
    RequestLogger::logWithContext(severity, message, metrics)

#define LOG(severity, message) \
    RequestLogger::logWithContext(severity, message, nullptr)

#endif //MONICAIMAGEPROCESS_LOGGINGUTILS_H