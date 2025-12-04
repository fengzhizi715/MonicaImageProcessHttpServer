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
#include "../../include/server/Server.h"
#include "../utils/aixlog.hpp"
#include "../utils/json.hpp"

namespace po = boost::program_options;
namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;


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

void print_config(Config config) {
    auto config_json = ordered_json::object();

    config_json["workers"] = config.num_threads;
    config_json["model_dir"] = config.model_dir;
    config_json["http_port"] = config.http_port;
    config_json["request_payload_limit"] = config.request_payload_limit;

    config_json["log"] = json::object();
    config_json["log"]["level"] = config.log_level;
    config_json["log"]["file"] = config.log_file;
    config_json["log"]["access_file"] = config.access_log_file;

    PLOG(L_INFO) << "Config values:\n" << config_json.dump(2) << std::endl;
}

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

    config.num_threads = numThreads;
    config.model_dir = modelPath;
    config.http_port = port;
    config.request_payload_limit = maxBodySize;

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

    print_config(config);

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