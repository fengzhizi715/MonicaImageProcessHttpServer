//
// Created by Tony on 2025/12/6.
// Router implementation for HTTP request handling
//

#include "../../include/server/Router.h"
#include <algorithm>

void Router::addRoute(const std::string& method, const std::string& path, RouteHandler handler) {
    routes_.emplace_back(method, path, handler);
}

void Router::addRoute(const std::string& method, const std::string& path, RouteHandler handler, const std::vector<Middleware>& middlewares) {
    routes_.emplace_back(method, path, handler, middlewares);
}

void Router::addMiddleware(Middleware middleware) {
    global_middlewares_.push_back(middleware);
}

bool Router::handleRequest(const std::string& method, const std::string& path) const {
    // Execute global middlewares first
    for (const auto& middleware : global_middlewares_) {
        if (!middleware()) {
            return true; // Middleware handled the request
        }
    }
    
    // Find matching route
    for (const auto& route : routes_) {
        if (route.method == method && matchPath(route.path, path)) {
            // Execute route-specific middlewares
            for (const auto& middleware : route.middlewares) {
                if (!middleware()) {
                    return true; // Middleware handled the request
                }
            }
            
            // Execute the handler
            route.handler();
            return true;
        }
    }
    
    // No matching route found
    return false;
}

bool Router::matchPath(const std::string& routePath, const std::string& requestPath) const {
    // Exact match
    if (routePath == requestPath) {
        return true;
    }
    
    // Prefix match for paths with query parameters (e.g., /api/cartoon?type=1)
    // Check if requestPath starts with routePath
    if (requestPath.find(routePath) == 0) {
        size_t routeLen = routePath.length();
        size_t requestLen = requestPath.length();
        
        // If requestPath is exactly the same as routePath, it's already matched above
        // If requestPath is longer, check if the next character is '?' (query params)
        if (requestLen > routeLen) {
            char nextChar = requestPath[routeLen];
            return (nextChar == '?');
        }
    }
    
    return false;
}