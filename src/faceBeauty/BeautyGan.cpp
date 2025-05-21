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

void BeautyGan::preprocess1(Mat image, Size& origin_size)
{
    origin_size = image.size();  // 保存原始尺寸
    cv::resize(image, image, cv::Size(256, 256));
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    std::vector<float> result(3 * 256 * 256);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(result.data() + i * 256 * 256, channels[i].data, 256 * 256 * sizeof(float));
    }

    this->input_image_1 = result;
}

void BeautyGan::preprocess2(Mat image, Size& origin_size)
{
    origin_size = image.size();  // 保存原始尺寸
    cv::resize(image, image, cv::Size(256, 256));
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    std::vector<float> result(3 * 256 * 256);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(result.data() + i * 256 * 256, channels[i].data, 256 * 256 * sizeof(float));
    }

    this->input_image_2 = result;
}

cv::Mat BeautyGan::postprocess(float* output_data) {
    std::vector<cv::Mat> output_channels;
    for (int i = 0; i < 3; ++i) {
        output_channels.emplace_back(256, 256, CV_32F, output_data + i * 256 * 256);
    }

    cv::Mat output_img;
    cv::merge(output_channels, output_img);
    output_img = output_img * 255.0;
    output_img.convertTo(output_img, CV_8U);
    return output_img;
}


void BeautyGan::inferImage(Mat& src, Mat makeup, Mat& dst) {

    cv::Size orig_sizeA;
    cv::Size orig_sizeB;

    // 图像预处理
    this->preprocess1(src, orig_sizeA);     // 原始人脸图像
    this->preprocess2(src, orig_sizeA);     // 参考妆容图像

    std::array<int64_t,4> input_shape {1,3,this->inpHeight, this->inpWidth};

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    vector<Value> ort_inputs;

    ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_1.data(), input_image_1.size(), input_shape.data(), input_shape.size()));
    ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_2.data(), input_image_2.size(), input_shape.data(), input_shape.size()));
    vector<Value> ort_outputs = this -> forward(ort_inputs);

    // 后处理
    float* output_data = ort_outputs.front().GetTensorMutableData<float>();

    cv::Mat beautygan_crop = postprocess(output_data);

    cv::resize(beautygan_crop, dst, orig_sizeA);
}