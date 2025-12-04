//
// Created by Tony on 2025/12/3.
// Enhanced logging utilities with request tracking and performance metrics
//

#include "../../include/server/LoggingUtils.h"
#include "../utils/aixlog.hpp"
#include <sstream>
#include <iomanip>

// Static member definitions
std::atomic<uint64_t> RequestLogger::requestCounter{0};
std::unordered_map<std::string, std::shared_ptr<RequestLogger::RequestMetrics>> RequestLogger::activeRequests;
std::mutex RequestLogger::requestsMutex;
boost::uuids::random_generator RequestLogger::uuidGenerator;

std::string RequestLogger::generateRequestId() {
    // Generate a UUID-based request ID
    boost::uuids::uuid uuid = uuidGenerator();
    std::string uuidStr = boost::uuids::to_string(uuid);
    
    // Remove hyphens for a cleaner ID
    uuidStr.erase(std::remove(uuidStr.begin(), uuidStr.end(), '-'), uuidStr.end());
    
    // Take only first 16 characters for brevity
    if (uuidStr.length() > 16) {
        uuidStr = uuidStr.substr(0, 16);
    }
    
    return uuidStr;
}

std::shared_ptr<RequestLogger::RequestMetrics> RequestLogger::startRequest(const std::string& endpoint) {
    std::string requestId = generateRequestId();
    auto metrics = std::make_shared<RequestMetrics>(requestId, endpoint);
    
    // Store in active requests map
    {
        std::lock_guard<std::mutex> lock(requestsMutex);
        activeRequests[requestId] = metrics;
    }
    
    // Log request start
    logWithContext(AixLog::Severity::info, 
                   "Starting request " + requestId + " for endpoint " + endpoint, 
                   metrics);
    
    return metrics;
}

void RequestLogger::endRequest(const std::shared_ptr<RequestMetrics>& metrics, int statusCode, size_t responseSize) {
    if (!metrics) return;
    
    metrics->endTime = std::chrono::high_resolution_clock::now();
    metrics->statusCode = statusCode;
    metrics->responseSize = responseSize;
    metrics->completed = true;
    
    // Remove from active requests
    {
        std::lock_guard<std::mutex> lock(requestsMutex);
        activeRequests.erase(metrics->requestId);
    }
    
    // Log performance metrics
    logRequestPerformance(metrics);
}

void RequestLogger::logWithContext(AixLog::Severity severity, const std::string& message, 
                                  const std::shared_ptr<RequestMetrics>& metrics) {
    std::stringstream logMessage;
    
    if (metrics) {
        logMessage << "[REQ:" << metrics->requestId << "] ";
    }
    
    logMessage << message;
    
    // Use the existing AixLog system
    switch (severity) {
        case AixLog::Severity::trace:
            PLOG(L_TRACE) << logMessage.str();
            break;
        case AixLog::Severity::debug:
            PLOG(L_DEBUG) << logMessage.str();
            break;
        case AixLog::Severity::info:
            PLOG(L_INFO) << logMessage.str();
            break;
        case AixLog::Severity::notice:
            PLOG(L_NOTICE) << logMessage.str();
            break;
        case AixLog::Severity::warning:
            PLOG(L_WARNING) << logMessage.str();
            break;
        case AixLog::Severity::error:
            PLOG(L_ERROR) << logMessage.str();
            break;
        case AixLog::Severity::fatal:
            PLOG(L_FATAL) << logMessage.str();
            break;
    }
}

std::string RequestLogger::getPerformanceStats() {
    std::lock_guard<std::mutex> lock(requestsMutex);
    
    std::stringstream stats;
    stats << "Active Requests: " << activeRequests.size();
    
    // TODO: Add more detailed statistics like average response time, etc.
    
    return stats.str();
}

void RequestLogger::logRequestPerformance(const std::shared_ptr<RequestMetrics>& metrics) {
    if (!metrics) return;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        metrics->endTime - metrics->startTime);
    
    std::stringstream perfMessage;
    perfMessage << "Request " << metrics->requestId 
                << " completed with status " << metrics->statusCode
                << " in " << duration.count() << "μs"
                << ", response size: " << metrics->responseSize << " bytes";
    
    // Log as INFO for successful requests, WARNING for client errors, ERROR for server errors
    AixLog::Severity severity = AixLog::Severity::info;
    if (metrics->statusCode >= 400 && metrics->statusCode < 500) {
        severity = AixLog::Severity::warning;
    } else if (metrics->statusCode >= 500) {
        severity = AixLog::Severity::error;
    }
    
    logWithContext(severity, perfMessage.str(), metrics);
}