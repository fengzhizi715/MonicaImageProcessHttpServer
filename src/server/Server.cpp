//
// Created by Tony on 2025/4/28.
//
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/url/url.hpp>
#include <boost/url/url_view.hpp>
#include <boost/url/parse.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../../include/server/Server.h"
#include "../../include/server/GlobalResource.h"
#include "../../include/server/HttpUtils.h"
#include "../../include/server/Config.h"
#include "../../include/server/LoggingUtils.h"
#include "../utils/json.hpp"
#include "../utils/aixlog.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;
using cv::Mat;

#define CONTENT_TYPE_PLAIN_TEXT "text/plain"
#define CONTENT_TYPE_IMAGE_JPEG "image/jpeg"
#define CONTENT_TYPE_JSON "application/json"

// session constructor
session::session(tcp::socket socket, std::shared_ptr<GlobalResource> globalResource, size_t maxBodySize)
    : socket_(std::move(socket)), globalResource_(globalResource), maxBodySize_(maxBodySize), requestMetrics_(nullptr) {}

// session start method
void session::start() { 
    // Initialize request tracking
    requestMetrics_ = RequestLogger::startRequest("unknown");
    do_read(); 
}

// Implementation of session class methods
void session::do_read() {
    auto self = shared_from_this();

    // Create a request parser with dynamic body
    auto parser = std::make_shared<http::request_parser<http::dynamic_body>>();
    // Set maximum allowed message body size
    parser->body_limit(maxBodySize_);

    // Asynchronously read the request
    http::async_read(socket_, buffer_, *parser,
                     [self, parser](beast::error_code ec, std::size_t bytes_transferred) {
                         if (ec) {
                             // Error handling: output error message and return response
                             std::string errorMsg = "Read error: " + ec.message();
                             LOG_REQ(AixLog::Severity::error, self->requestMetrics_, errorMsg);
                             
                             http::response<http::string_body> res{http::status::bad_request, self->req_.version()};
                             res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
                             res.body() = "Error reading request: " + ec.message();
                             res.prepare_payload();
                             self->do_write(res);
                             return;
                         }
                         // Get the request from the parser
                         self->req_ = parser->release();
                         
                         // Update request metrics with actual endpoint
                         if (self->requestMetrics_) {
                             self->requestMetrics_->endpoint = std::string(self->req_.target());
                             self->requestMetrics_->requestSize = bytes_transferred;
                         }
                         
                         self->handle_request();
                     }
    );
}

void session::send_error_response(const http::status status, const std::string& message) {
    LOG_REQ(AixLog::Severity::warning, requestMetrics_, "Sending error response: " + message);
    
    http::response<http::string_body> res{status, req_.version()};
    res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
    res.body() = message;
    res.prepare_payload();
    do_write(res);
}

void session::handle_request() {
    try {
        auto target = std::string(req_.target());
        
        // Update endpoint in metrics
        if (requestMetrics_) {
            requestMetrics_->endpoint = target;
        }

        if (target == "/health") {
            LOG_REQ(AixLog::Severity::info, requestMetrics_, "Health check request received");
            
            http::response<http::string_body> res{http::status::ok, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
            res.body() = "OK";
            res.prepare_payload();
            
            // End request tracking
            if (requestMetrics_) {
                RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
            }
            
            return do_write(res);
        }

        if (target == "/version") {
            LOG_REQ(AixLog::Severity::info, requestMetrics_, "Version request received");
            
            auto json_string = json::object({
                {"onnxruntime_version", OrtGetApiBase()->GetVersionString() },
                {"server_version","1.0.0"}}).dump();

            http::response<http::string_body> res{http::status::ok, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_JSON);
            res.body() = json_string;
            res.prepare_payload();
            
            // End request tracking
            if (requestMetrics_) {
                RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
            }
            
            return do_write(res);
        }

        auto method = req_.method();
        // Simple routing: dispatch different logic based on target
        if (method == http::verb::post) {
            if (target == "/api/sketchDrawing") {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Sketch drawing request received");
                process_image(target, [this](Mat src) { return globalResource_->processSketchDrawing(src); });
            } else if (target == "/api/faceDetect") {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Face detection request received");
                process_image(target, [this](Mat src) { return globalResource_->processFaceDetect(src); });
            } else if (target == "/api/faceLandMark") {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Face landmark request received");
                process_image(target, [this](Mat src) { return globalResource_->processFaceLandMark(src); });
            } else if (target.find("/api/faceSwap") == 0) {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Face swap request received");
                try {
                    // Parse path + query parameters using parse_relative_ref
                    auto targetRes = boost::urls::parse_relative_ref(target);
                    if (!targetRes) {
                        throw std::runtime_error("Failed to parse URL: " + targetRes.error().message());
                    }

                    boost::urls::url_view url_view = targetRes.value();
                    std::string status_param = "false";  // Default value
                    // Correct parameter retrieval method
                    auto params = url_view.params();
                    if (auto it = params.find("status"); it != params.end()) {
                        auto value = (*it).value;
                        status_param = std::string(value.data(), value.size());
                    }
                    bool status = (status_param == "true");

                    // Parse multipart/form-data
                    auto parts = parseMultipartFormDataManual(req_);
                    if (parts.find("src") == parts.end() || parts.find("target") == parts.end()) {
                        throw std::runtime_error("Missing images in request. Expected 'src' and 'target' fields.");
                    }

                    Mat src = binaryToCvMat(parts["src"]);
                    Mat target_img = binaryToCvMat(parts["target"]);
                    
                    if (src.empty() || target_img.empty()) {
                        throw std::runtime_error("Invalid image data: decoded image is empty.");
                    }
                    
                    Mat dst = globalResource_->processFaceSwap(src, target_img, status);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    
                    // End request tracking
                    if (requestMetrics_) {
                        RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
                    }
                    
                    do_write(res);
                } catch (const std::exception& e) {
                    LOG_REQ(AixLog::Severity::error, requestMetrics_, "Error processing face swap: " + std::string(e.what()));
                    send_error_response(http::status::bad_request, "Error processing face swap: " + std::string(e.what()));
                }
            } else if (target.find("/api/cartoon") == 0) {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Cartoon request received");
                try {
                    // Parse path + query parameters using parse_relative_ref
                    auto targetRes = boost::urls::parse_relative_ref(target);
                    if (!targetRes) {
                        throw std::runtime_error("Failed to parse URL: " + targetRes.error().message());
                    }

                    boost::urls::url_view url_view = targetRes.value();
                    int type_param = 0;  // Default value
                    // Correct parameter retrieval method
                    auto params = url_view.params();
                    if (auto it = params.find("type"); it != params.end()) {
                        try {
                            auto value = (*it).value;
                            type_param = std::stoi(std::string(value.data(), value.size()));
                        } catch (const std::invalid_argument&) {
                            throw std::runtime_error("Invalid type parameter. Expected integer.");
                        }
                    }

                    Mat src = requestBodyToCvMat(req_);
                    if (src.empty()) {
                        throw std::runtime_error("Invalid image data: decoded image is empty.");
                    }
                    
                    Mat dst = globalResource_->processCartoon(src, type_param);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    
                    // End request tracking
                    if (requestMetrics_) {
                        RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
                    }
                    
                    do_write(res);
                } catch (const std::exception& e) {
                    LOG_REQ(AixLog::Severity::error, requestMetrics_, "Error processing cartoon: " + std::string(e.what()));
                    send_error_response(http::status::bad_request, "Error processing cartoon: " + std::string(e.what()));
                }
            } else if (target == "/api/faceBeauty") {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Face beauty request received");
                try {
                    // Parse multipart/form-data
                    auto parts = parseMultipartFormDataManual(req_);
                    if (parts.find("src") == parts.end() || parts.find("makeup") == parts.end()) {
                        throw std::runtime_error("Missing images in request.");
                    }

                    Mat src = binaryToCvMat(parts["src"]);
                    Mat makeup = binaryToCvMat(parts["makeup"]);
                    Mat dst = globalResource_->processBeauty(src, makeup);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    
                    // End request tracking
                    if (requestMetrics_) {
                        RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
                    }
                    
                    do_write(res);
                } catch (const std::exception& e) {
                    LOG_REQ(AixLog::Severity::error, requestMetrics_, "Error processing face beauty: " + std::string(e.what()));
                    send_error_response(http::status::bad_request, "Error processing face beauty: " + std::string(e.what()));
                }
            } else if (target == "/api/changePersonBackground") {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Change person background request received");
                try {
                    // Parse multipart/form-data
                    auto parts = parseMultipartFormDataManual(req_);
                    if (parts.find("src") == parts.end() || parts.find("background") == parts.end()) {
                        throw std::runtime_error("Missing images in request.");
                    }

                    Mat src = binaryToCvMat(parts["src"]);
                    Mat background = binaryToCvMat(parts["background"]);
                    Mat dst = globalResource_->processPersonBackground(src, background);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    
                    // End request tracking
                    if (requestMetrics_) {
                        RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
                    }
                    
                    do_write(res);
                } catch (const std::exception& e) {
                    LOG_REQ(AixLog::Severity::error, requestMetrics_, "Error processing person background: " + std::string(e.what()));
                    send_error_response(http::status::bad_request, "Error processing person background: " + std::string(e.what()));
                }
            } else if (target.find("/api/changeHairColor") == 0) {
                LOG_REQ(AixLog::Severity::info, requestMetrics_, "Change hair color request received");
                try {
                    // Parse path + query parameters using parse_relative_ref
                    auto targetRes = boost::urls::parse_relative_ref(target);
                    if (!targetRes) {
                        throw std::runtime_error("Failed to parse URL: " + targetRes.error().message());
                    }

                    boost::urls::url_view url_view = targetRes.value();
                    int hue_param = 0;  // Default value
                    // Correct parameter retrieval method
                    auto params = url_view.params();
                    if (auto it = params.find("targetHue"); it != params.end()) {
                        try {
                            auto value = (*it).value;
                            hue_param = std::stoi(std::string(value.data(), value.size()));
                        } catch (const std::invalid_argument&) {
                            throw std::runtime_error("Invalid targetHue parameter. Expected integer.");
                        }
                    } else {
                        throw std::runtime_error("Missing required parameter: targetHue");
                    }

                    Mat src = requestBodyToCvMat(req_);
                    if (src.empty()) {
                        throw std::runtime_error("Invalid image data: decoded image is empty.");
                    }
                    
                    Mat dst = globalResource_->changeHairColor(src, hue_param, 1.3f);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    
                    // End request tracking
                    if (requestMetrics_) {
                        RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
                    }
                    
                    do_write(res);
                } catch (const std::exception& e) {
                    LOG_REQ(AixLog::Severity::error, requestMetrics_, "Error processing hair color change: " + std::string(e.what()));
                    send_error_response(http::status::bad_request, "Error processing hair color change: " + std::string(e.what()));
                }
            } else {
                // Other interfaces return 404
                LOG_REQ(AixLog::Severity::warning, requestMetrics_, "Endpoint not found: " + target);
                http::response<http::string_body> res{http::status::not_found, req_.version()};
                res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
                res.body() = "Not Found";
                res.prepare_payload();
                
                // End request tracking
                if (requestMetrics_) {
                    RequestLogger::endRequest(requestMetrics_, 404, res.body().length());
                }
                
                do_write(res);
            }
        } else {
            // Other interfaces return 404
            LOG_REQ(AixLog::Severity::warning, requestMetrics_, "Method not allowed for endpoint: " + target);
            http::response<http::string_body> res{http::status::not_found, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
            res.body() = "Not Found";
            res.prepare_payload();
            
            // End request tracking
            if (requestMetrics_) {
                RequestLogger::endRequest(requestMetrics_, 404, res.body().length());
            }
            
            do_write(res);
        }
    } catch (const std::exception& e) {
        LOG_REQ(AixLog::Severity::error, requestMetrics_, "Unexpected error in handle_request: " + std::string(e.what()));
        send_error_response(http::status::internal_server_error, "Internal server error: " + std::string(e.what()));
    } catch (...) {
        LOG_REQ(AixLog::Severity::error, requestMetrics_, "Unknown error in handle_request");
        send_error_response(http::status::internal_server_error, "Unknown internal server error");
    }
}

void session::process_image(const std::string& target, std::function<Mat(Mat)> processor) {
    try {
        Mat src = requestBodyToCvMat(req_);
        Mat dst = processor(src);
        std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

        http::response<http::string_body> res{http::status::ok, req_.version()};
        res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
        res.body() = std::move(encodedImage);
        res.prepare_payload();
        
        // End request tracking
        if (requestMetrics_) {
            RequestLogger::endRequest(requestMetrics_, 200, res.body().length());
        }
        
        do_write(res);
    } catch (const std::exception& e) {
        http::response<http::string_body> res{http::status::internal_server_error, req_.version()};
        res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
        res.body() = "Error: " + std::string(e.what());
        res.prepare_payload();
        
        // End request tracking
        if (requestMetrics_) {
            RequestLogger::endRequest(requestMetrics_, 500, res.body().length());
        }
        
        do_write(res);
    }
}

template<class Response>
void session::do_write(Response& res) {
    auto self = shared_from_this();
    auto sp = std::make_shared<Response>(std::move(res));
    http::async_write(socket_, *sp,
                      [self, sp](beast::error_code ec, std::size_t) {
                          self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                      });
}

void server::run() {
    LOG(AixLog::Severity::info, "Server starting to accept connections");
    do_accept();
}

// server constructor
server::server(net::io_context& ioc, tcp::endpoint endpoint, std::string modelPath, size_t maxBodySize)
    : acceptor_(ioc),
      globalResource_(std::make_shared<GlobalResource>(modelPath)),
      maxBodySize_(maxBodySize) {
    beast::error_code ec;
    acceptor_.open(endpoint.protocol(), ec);
    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    acceptor_.set_option(net::socket_base::receive_buffer_size(1024 * 1024), ec);
    acceptor_.bind(endpoint, ec);
    acceptor_.listen(net::socket_base::max_listen_connections, ec);
    
    LOG(AixLog::Severity::info, "Server initialized on port " + std::to_string(endpoint.port()));
}

void server::do_accept() {
    acceptor_.async_accept(
        [this](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                LOG(AixLog::Severity::info, "New connection accepted");
                // Pass global resources to session
                std::make_shared<session>(std::move(socket), globalResource_, maxBodySize_)->start();
            } else {
                LOG(AixLog::Severity::warning, "Accept error: " + ec.message());
            }
            do_accept();
        });
}
