//
// Created by Tony on 2025/3/25.
//
#include <boost/program_options.hpp>
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
#include "../../include/server/GlobalResource.h"
#include "../../include/server/HttpUtils.h"
#include "../../include/server/Config.h"
#include "../utils/aixlog.hpp"

namespace po = boost::program_options;
namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

#define CONTENT_TYPE_PLAIN_TEXT "text/plain"
#define CONTENT_TYPE_IMAGE_JPEG "image/jpeg"

// 用于处理单个 HTTP 会话
class session : public std::enable_shared_from_this<session> {
public:
    session(tcp::socket socket, std::shared_ptr<GlobalResource> globalResource, size_t maxBodySize)
            : socket_(std::move(socket)), globalResource_(globalResource), maxBodySize_(maxBodySize) {}

    void start() { do_read(); }

private:
    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::dynamic_body> req_;
    // 引用全局资源
    std::shared_ptr<GlobalResource> globalResource_;
    size_t maxBodySize_;

    void do_read() {
        auto self = shared_from_this();

        // 创建一个请求解析器，使用 dynamic_body 类型
        auto parser = std::make_shared<http::request_parser<http::dynamic_body>>();
        // 设置允许的最大消息体大小
        parser->body_limit(maxBodySize_);

        // 异步读取请求
        http::async_read(socket_, buffer_, *parser,
                         [self, parser](beast::error_code ec, std::size_t bytes_transferred) {
                             if (ec) {
                                 // 错误处理：输出错误信息并返回响应
                                 std::cerr << "Read error: " << ec.message() << std::endl;
                                 http::response<http::string_body> res{http::status::bad_request, self->req_.version()};
                                 res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
                                 res.body() = "Error reading request: " + ec.message();
                                 res.prepare_payload();
                                 self->do_write(res);
                                 return;
                             }
                             // 从解析器中获取请求
                             self->req_ = parser->release();
                             self->handle_request();
                         }
        );
    }

    // 根据请求路径处理请求
    void handle_request() {
        auto target = std::string(req_.target());

        if (target == "/health") {
            http::response<http::string_body> res{http::status::ok, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
            res.body() = "OK";
            res.prepare_payload();
            return do_write(res);
        }

        auto method = req_.method();
        // 简单路由：根据 target 分发不同的逻辑
        if (method == http::verb::post) {
            if (target == "/api/sketchDrawing") {
                process_image(target, [this](Mat src) { return globalResource_->processSketchDrawing(src); });
            } else if (target == "/api/faceDetect") {
                process_image(target, [this](Mat src) { return globalResource_->processFaceDetect(src); });
            } else if (target == "/api/faceLandMark") {
                process_image(target, [this](Mat src) { return globalResource_->processFaceLandMark(src); });
            } else if (target.find("/api/faceSwap") == 0) {
                try {
                    // 使用 parse_relative_ref 解析路径 + 查询参数
                    auto targetRes = boost::urls::parse_relative_ref(target);
                    if (!targetRes) {
                        std::cerr << "URL parsing failed: " << targetRes.error().message() << std::endl;
                        return;
                    }

                    boost::urls::url_view url_view = targetRes.value();
                    std::string status_param = "false";  // 默认值
                    // 正确的参数获取方式
                    auto params = url_view.params();
                    if (auto it = params.find("status"); it != params.end()) {
                        auto value = (*it).value;  // 注意：这里可能是 .value 而不是 .value()
                        status_param = std::string(value.data(), value.size());
                    }
                    bool status = (status_param == "true");
                    cout << "status = "<< status << endl;

                    // 解析 multipart/form-data
                    auto parts = parseMultipartFormDataManual(req_);
                    if (parts.find("src") == parts.end() || parts.find("target") == parts.end()) {
                        throw std::runtime_error("Missing images in request.");
                    }

                    Mat src = binaryToCvMat(parts["src"]);
                    Mat target = binaryToCvMat(parts["target"]);
                    Mat dst = globalResource_.get()->processFaceSwap(src, target, status);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    do_write(res);
                } catch (const std::exception& e) {
                    http::response<http::string_body> res{http::status::bad_request, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
                    res.body() = "Error processing face swap: " + std::string(e.what());
                    res.prepare_payload();
                    do_write(res);
                }
            } else if (target.find("/api/cartoon") == 0) {
                try {
                    // 使用 parse_relative_ref 解析路径 + 查询参数
                    auto targetRes = boost::urls::parse_relative_ref(target);
                    if (!targetRes) {
                        std::cerr << "URL parsing failed: " << targetRes.error().message() << std::endl;
                        return;
                    }

                    boost::urls::url_view url_view = targetRes.value();
                    int type_param = 0;  // 默认值
                    // 正确的参数获取方式
                    auto params = url_view.params();
                    if (auto it = params.find("type"); it != params.end()) {
                        auto value = (*it).value;
                        type_param = std::stoi(value);
                    }

                    Mat src = requestBodyToCvMat(req_);
                    Mat dst = globalResource_->processCartoon(src, type_param);
                    std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

                    http::response<http::string_body> res{http::status::ok, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
                    res.body() = std::move(encodedImage);
                    res.prepare_payload();
                    do_write(res);
                } catch (const std::exception& e) {
                    http::response<http::string_body> res{http::status::bad_request, req_.version()};
                    res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
                    res.body() = "Error processing face swap: " + std::string(e.what());
                    res.prepare_payload();
                    do_write(res);
                }
            }
    } else {
            // 其他接口返回 404
            http::response<http::string_body> res{http::status::not_found, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
            res.body() = "Not Found";
            res.prepare_payload();
            do_write(res);
        }
    }

    void process_image(const std::string& target, std::function<Mat(Mat)> processor) {
        try {
            Mat src = requestBodyToCvMat(req_);
            Mat dst = processor(src);
            std::string encodedImage = cvMatToResponseBody(dst, ".jpg");

            http::response<http::string_body> res{http::status::ok, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_IMAGE_JPEG);
            res.body() = std::move(encodedImage);
            res.prepare_payload();
            do_write(res);
        } catch (const std::exception& e) {
            http::response<http::string_body> res{http::status::internal_server_error, req_.version()};
            res.set(http::field::content_type, CONTENT_TYPE_PLAIN_TEXT);
            res.body() = "Error: " + std::string(e.what());
            res.prepare_payload();
            do_write(res);
        }
    }


    template<class Response>
    void do_write(Response& res) {
        auto self = shared_from_this();
        auto sp = std::make_shared<Response>(std::move(res));
        http::async_write(socket_, *sp,
                          [self, sp](beast::error_code ec, std::size_t) {
                              self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                          });
    }
};

// HTTP 服务器：监听指定端口，并为每个连接创建一个 session
class server {
public:
    server(net::io_context& ioc, tcp::endpoint endpoint, std::string modelPath, size_t maxBodySize)
            : acceptor_(ioc)
            , globalResource_(std::make_shared<GlobalResource>(modelPath)) // 全局资源初始化，只调用一次
            , maxBodySize_(maxBodySize) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        acceptor_.set_option(net::socket_base::receive_buffer_size(1024 * 1024), ec);
        acceptor_.bind(endpoint, ec);
        acceptor_.listen(net::socket_base::max_listen_connections, ec);
    }

    void run() {
        do_accept();
    }

private:
    tcp::acceptor acceptor_;
    std::shared_ptr<GlobalResource> globalResource_;
    size_t maxBodySize_;

    void do_accept() {
        acceptor_.async_accept(
                [this](beast::error_code ec, tcp::socket socket) {
                    if (!ec) {
                        // 将全局资源传递给 session
                        std::make_shared<session>(std::move(socket), globalResource_, maxBodySize_)->start();
                    }
                    do_accept();
                });
    }
};

class SinkCoutWithFilter : public AixLog::SinkCout {
    AixLog::Filter _allow;
    AixLog::Filter _deny;

public:
    SinkCoutWithFilter(
            const AixLog::Filter &allow, const AixLog::Filter &deny,
            const std::string &format = "%Y-%m-%d %H:%M:%S.#ms [#severity] (#tag_func)"
    )
            : SinkCout(AixLog::Filter(), format), _allow(allow), _deny(deny) {
    }

    void log(const AixLog::Metadata &metadata, const std::string &message) override {
        if ((_allow.is_empty() || _allow.match(metadata)) && (_deny.is_empty() || !_deny.match(metadata)))
            SinkCout::log(metadata, message);
    }
};

class SinkFileWithFilter : public AixLog::SinkFile {
    AixLog::Filter _allow;
    AixLog::Filter _deny;

public:
    SinkFileWithFilter(
            const AixLog::Filter &allow, const AixLog::Filter &deny, const std::string &filename,
            const std::string &format = "%Y-%m-%d %H:%M:%S.#ms [#severity] (#tag_func)"
    )
            : SinkFile(AixLog::Filter(), filename, format), _allow(allow), _deny(deny) {
    }

    void log(const AixLog::Metadata &metadata, const std::string &message) override {
        if ((_allow.is_empty() || _allow.match(metadata)) && (_deny.is_empty() || !_deny.match(metadata)))
            SinkFile::log(metadata, message);
    }
};

int main(int argc, char* argv[]) {

    Config config;
    // 默认配置参数
    int port = 8080;
    int numThreads = std::thread::hardware_concurrency();
    std::string modelPath = "/Users/Tony/CLionProjects/MonicaImageProcessHttpServer/models";
    size_t maxBodySize = 10 * 1024 * 1024; // 默认最大请求体大小为 10 MB

    // 定义命令行选项
    po::options_description desc("Allowed options", 200);
    desc.add_options()
            ("help,h", "Display help message")
            ("http-port,p", po::value<int>(&port)->default_value(8080), "HTTP server port")
            ("num-threads,t", po::value<int>(&numThreads)->default_value(std::thread::hardware_concurrency()), "Number of worker threads")
            ("model-dir,m", po::value<std::string>(&modelPath)->default_value(modelPath), "Path to the model directory")
            ("max-body-size,b", po::value<size_t>(&maxBodySize)->default_value(maxBodySize), "Maximum HTTP body size in bytes")
            ("log-level", po::value<std::string>()->default_value("info"),"Log level(debug, info, warn, error, fatal).\nDefault: info")
            ("log-file", po::value<std::string>()->default_value(""),"Log file path.If not specified, logs will be printed to stdout.")
            ("access-log-file", po::value<std::string>()->default_value(""),"Access log file path.\nIf not specified, logs will be printed to stdout.");

    // 解析命令行参数
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    config.log_level = vm["log-level"].as<std::string>();
    config.log_file = vm["log-file"].as<std::string>();
    config.access_log_file = vm["access-log-file"].as<std::string>();

    std::map<AixLog::Severity, std::string> log_level_map = {
            {AixLog::Severity::debug, "debug"}, {AixLog::Severity::info, "info"},	{AixLog::Severity::warning, "warn"},
            {AixLog::Severity::error, "error"}, {AixLog::Severity::fatal, "fatal"},
    };
    AixLog::Severity log_level = AixLog::Severity::info;
    for (auto &level : log_level_map) {
        if (config.log_level == level.second) {
            log_level = level.first;
            break;
        }
    }

    std::shared_ptr<AixLog::Sink> log_file;
    std::shared_ptr<AixLog::Sink> log_access_file;
    AixLog::Filter for_access;
    for_access.add_filter("ACCESS", AixLog::Severity::info);

    if (config.log_file.empty())
        log_file = std::make_shared<SinkCoutWithFilter>(log_level, for_access);
    else
        log_file = std::make_shared<SinkFileWithFilter>(log_level, for_access, config.log_file);

    if (config.access_log_file.empty())
        log_access_file = std::make_shared<SinkCoutWithFilter>(for_access, AixLog::Filter());
    else
        log_access_file = std::make_shared<SinkFileWithFilter>(
                for_access, AixLog::Filter(), config.access_log_file, "%Y-%m-%d %H-%M-%S.#ms [#severity]"
        );

    AixLog::Log::init({log_file, log_access_file});

    net::io_context ioc{numThreads};
    tcp::endpoint endpoint{tcp::v4(), static_cast<unsigned short>(port)};
    server srv(ioc, endpoint, modelPath, maxBodySize);
    srv.run();

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads - 1; ++i) {
        threads.emplace_back([&ioc](){ ioc.run(); });
    }
    ioc.run();

    for (auto& t : threads)
        t.join();

    return 0;
}