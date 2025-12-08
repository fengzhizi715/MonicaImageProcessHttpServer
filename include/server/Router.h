//
// Created by Tony on 2025/12/6.
// Router for HTTP request handling with route mapping and middleware support
//

#ifndef MONICAIMAGEPROCESS_ROUTER_H
#define MONICAIMAGEPROCESS_ROUTER_H

#include <functional>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <boost/beast/http.hpp>
#include <boost/beast/core.hpp>

namespace beast = boost::beast;
namespace http = beast::http;

class GlobalResource;
class RequestLogger;

// Route handler function type
using RouteHandler = std::function<void()>;

// Middleware function type
using Middleware = std::function<bool()>; // Return true to continue, false to stop

struct Route {
    std::string method;
    std::string path;
    RouteHandler handler;
    std::vector<Middleware> middlewares;
    
    Route(const std::string& m, const std::string& p, RouteHandler h) 
        : method(m), path(p), handler(h) {}
        
    Route(const std::string& m, const std::string& p, RouteHandler h, const std::vector<Middleware>& mw) 
        : method(m), path(p), handler(h), middlewares(mw) {}
};

class Router {
private:
    std::vector<Route> routes_;
    std::vector<Middleware> global_middlewares_;

public:
    Router() = default;
    
    // Add a route
    void addRoute(const std::string& method, const std::string& path, RouteHandler handler);
    
    // Add a route with middleware
    void addRoute(const std::string& method, const std::string& path, RouteHandler handler, const std::vector<Middleware>& middlewares);
    
    // Add global middleware
    void addMiddleware(Middleware middleware);
    
    // Find and execute route handler
    bool handleRequest(const std::string& method, const std::string& path) const;
    
    // Match path with parameters (e.g., /api/cartoon?type=1)
    bool matchPath(const std::string& routePath, const std::string& requestPath) const;
};

#endif //MONICAIMAGEPROCESS_ROUTER_H