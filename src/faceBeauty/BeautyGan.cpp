//
// Created by Tony on 2025/5/21.
//
#include "../../include/faceBeauty/BeautyGan.h"

BeautyGan::BeautyGan(std::string modelPath, const char* logId, const char* provider): OnnxRuntimeBase(modelPath, logId, provider)
{
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->outHeight = output_node_dims[0][2];
    this->outWidth = output_node_dims[0][3];
}

vector<float> BeautyGan::preprocess(const Mat& image)
{
    Mat resized;
    cv::resize(image, resized, cv::Size(this->inpWidth, this->inpHeight));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> result(this->inpWidth * this->inpHeight * 3);
    const unsigned int channel_step = inpHeight * inpWidth;
    for (int i = 0; i < 3; ++i) {
        std::memcpy(result.data() + i * channel_step, channels[i].data, channel_step * sizeof(float));
    }

    return result;
}

cv::Mat BeautyGan::postprocess(float* output_data) {
    std::vector<cv::Mat> output_channels;
    const unsigned int channel_step = outHeight * outWidth;
    for (int i = 0; i < 3; ++i) {
        output_channels.emplace_back(outHeight, outWidth, CV_32F, output_data + i * channel_step);
    }

    cv::Mat output_img;
    cv::merge(output_channels, output_img);
    output_img = output_img * 255.0;
    output_img.convertTo(output_img, CV_8U);
    return output_img;
}


void BeautyGan::inferImage(const Mat& src, const Mat& makeup, Mat& dst) {

    // 图像预处理
    this->input_image_1 = this->preprocess(src);        // 原始人脸图像
    this->input_image_2 = this->preprocess(makeup);     // 参考妆容图像

    std::array<int64_t,4> input_shape {1,3,this->inpHeight, this->inpWidth};

    vector<Value> ort_inputs;

    ort_inputs.push_back(Value::CreateTensor<float>(memory_info_handler, input_image_1.data(), input_image_1.size(), input_shape.data(), input_shape.size()));
    ort_inputs.push_back(Value::CreateTensor<float>(memory_info_handler, input_image_2.data(), input_image_2.size(), input_shape.data(), input_shape.size()));
    vector<Value> ort_outputs = this -> forward(ort_inputs);

    // 后处理
    float* output_data = ort_outputs.front().GetTensorMutableData<float>();
    cv::Mat beautygan_crop = this->postprocess(output_data);

    cv::resize(beautygan_crop, dst, src.size());
}