//
// Created by Tony on 2025/3/26.
//

#ifndef MONICAIMAGEPROCESS_HTTPUTILS_H
#define MONICAIMAGEPROCESS_HTTPUTILS_H

#include <boost/beast.hpp>
#include <boost/beast/http.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdint>

namespace beast = boost::beast;
namespace http = beast::http;

// Parse multipart/form-data and convert to map
std::map<std::string, std::vector<uint8_t>> parseMultipartFormDataManual(http::request<http::dynamic_body>& req);

// Convert binary data to Mat
cv::Mat binaryToCvMat(std::vector<uint8_t>& data);

// Convert HTTP request body to Mat object
cv::Mat requestBodyToCvMat(http::request<http::dynamic_body>& req);

// Convert Mat to std::string for HTTP response
std::string cvMatToResponseBody(cv::Mat& image, const std::string& extension);

#endif //MONICAIMAGEPROCESS_HTTPUTILS_H
