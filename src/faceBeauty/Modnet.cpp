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

void Modnet::inferImage(Mat& src, Mat& mask) {
    this->preprocess(src);
    std::array<int64_t,4> input_shape {1,3, this->inpHeight, this->inpWidth};

    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info_handler, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    vector<Value> ort_outputs = this -> forward(input_tensor_);

    auto output_data = ort_outputs[0].GetTensorMutableData<float>();

    cv::Mat alpha = this -> postprocess(output_data, this->inpWidth, this->inpHeight);  // CV_32FC1
    cv::resize(alpha, alpha, src.size());                // 缩放到原图尺寸
    alpha.convertTo(alpha, CV_32FC1);                    // 确保是 float32 类型

    mask = alpha;
}

void Modnet::changeBackground(Mat src, Mat background, Mat& dst) {
    Mat alpha;
    this->inferImage(src, alpha);

    cv::Mat alpha_3ch;
    cv::merge(std::vector<cv::Mat>{alpha, alpha, alpha}, alpha_3ch);
    alpha_3ch.convertTo(alpha_3ch, CV_32FC3);  // 转成 float32 三通道

    // 转换原图和背景为 float
    cv::Mat image_f, bg_f;
    src.convertTo(image_f, CV_32FC3, 1.0 / 255.0);

    if (background.empty()) {
        background = Mat(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    } else {
        cv::resize(background, background, src.size());
    }
    background.convertTo(bg_f, CV_32FC3, 1.0 / 255.0);

    // 计算 1 - alpha
    cv::Mat one_minus_alpha;
    cv::subtract(1.0, alpha_3ch, one_minus_alpha, cv::noArray(), CV_32FC3);

    // alpha blending
    cv::Mat fg_part = image_f.mul(alpha_3ch);           // 前景
    cv::Mat bg_part = bg_f.mul(one_minus_alpha);        // 背景
    cv::Mat blended_f;
    cv::add(fg_part, bg_part, blended_f);               // 混合结果（float）

    // 转回 8bit
    cv::Mat blended;
    blended_f.convertTo(blended, CV_8UC3, 255.0);
    dst = blended;
}