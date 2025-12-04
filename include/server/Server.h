//
// Created by Tony on 2025/4/28.
//

#ifndef MONICAIMAGEPROCESS_SERVER_H
#define MONICAIMAGEPROCESS_SERVER_H

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <memory>
#include <string>
#include "../utils/aixlog.hpp"
#include "LoggingUtils.h"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

class GlobalResource;
class RequestLogger;

// Forward declaration of session class
class session : public std::enable_shared_from_this<session> {
public:
    session(tcp::socket socket, std::shared_ptr<GlobalResource> globalResource, size_t maxBodySize);
    void start();

private:
    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::dynamic_body> req_;
    std::shared_ptr<GlobalResource> globalResource_;
    size_t maxBodySize_;
    std::shared_ptr<RequestLogger::RequestMetrics> requestMetrics_;

    void do_read();
    void handle_request();
    void process_image(const std::string& target, std::function<cv::Mat(cv::Mat)> processor);

    template<class Response>
    void do_write(Response& res);
    
    void send_error_response(const http::status status, const std::string& message);
};

// HTTP server class
class server {
public:
    server(net::io_context& ioc, tcp::endpoint endpoint, std::string modelPath, size_t maxBodySize);
    void run();

private:
    tcp::acceptor acceptor_;
    std::shared_ptr<GlobalResource> globalResource_;
    size_t maxBodySize_;

    void do_accept();
};

#endif // MONICAIMAGEPROCESS_SERVER_H
