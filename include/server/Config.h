//
// Created by Tony on 2025/4/24.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_CONFIG_H
#define MONICAIMAGEPROCESSHTTPSERVER_CONFIG_H

class Config {
public:
    short http_port = 80;

    std::string log_level;
    std::string log_file;
    std::string access_log_file;

    long num_threads = 4;
    std::string model_dir;
    long request_payload_limit = 1024 * 1024 * 10;
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_CONFIG_H