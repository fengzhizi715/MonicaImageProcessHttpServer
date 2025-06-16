//
// Created by Tony on 2025/6/16.
//
#include "../../include/faceBeauty/Modnet.h"

Modnet::Modnet(std::string modelPath, const char* logId, const char* provider): OnnxRuntimeBase(modelPath, logId, provider)
{
    this->inpHeight = 512;
    this->inpWidth = 512;
}

void Modnet::preprocess(const cv::Mat& image) {
    cv::Mat resized, float_img;
    cv::resize(image, resized, cv::Size(512, 512));
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    this->input_image_.resize(this->inpWidth * this->inpHeight * image.channels());

    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < this->inpHeight; ++h)
            for (int w = 0; w < this->inpWidth; ++w)
                this->input_image_[c * this->inpHeight * this->inpWidth + h * this->inpWidth + w] = float_img.at<cv::Vec3f>(h, w)[c];
}

// 假设 postprocess() 将 MODNet 输出 float* 转为 CV_32FC1 alpha mask（取值范围 0~1）
cv::Mat Modnet::postprocess(float* output_data, int width, int height) {
    cv::Mat alpha(height, width, CV_32FC1, output_data);
    // 拷贝一份，防止 output_data 被释放
    return alpha.clone();
}

void Modnet::inferImage(Mat& src, Mat& dst) {
    this->preprocess(src);
    std::array<int64_t,4> input_shape {1,3, this->inpHeight, this->inpWidth};

    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info_handler, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    vector<Value> ort_outputs = this -> forward(input_tensor_);

    auto output_data = ort_outputs[0].GetTensorMutableData<float>();

    cv::Mat alpha = this -> postprocess(output_data, this->inpWidth, this->inpHeight);  // CV_32FC1
    cv::resize(alpha, alpha, src.size());                // 缩放到原图尺寸
    alpha.convertTo(alpha, CV_32FC1);                    // 确保是 float32 类型

    dst = alpha;
}