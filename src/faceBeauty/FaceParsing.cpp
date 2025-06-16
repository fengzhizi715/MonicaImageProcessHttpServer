//
// Created by Tony on 2025/5/20.
//
#include "../../include/faceBeauty/FaceParsing.h"

FaceParsing::FaceParsing(std::string modelPath, const char* logId, const char* provider): OnnxRuntimeBase(modelPath, logId, provider)
{
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
}

void FaceParsing::preprocess(Mat src)
{
    Mat dst;
    cvtColor(src, dst, COLOR_BGR2RGB);
    resize(dst, dst, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);

    dst.convertTo(dst, CV_32F, 1.0f/255.0f); // 归一化 [0,1]

    // HWC->CHW 平铺
    this->input_image_.reserve(1*3*this->inpHeight*this->inpWidth);
    for(int c=0;c<3;++c)
        for(int h=0; h< this->inpHeight;++h)
            for(int w=0; w< this->inpWidth;++w)
                this->input_image_.push_back(dst.at<cv::Vec3f>(h,w)[c]);
}


// 解析 output_data 得到 label_map，便于未来可以提取更多比如 眼睛、嘴巴、鼻子 等等
Mat FaceParsing::getLabelMap(float* output_data, int num_classes, int height, int width) {
    Mat label_map(height, width, CV_8UC1);
    const unsigned int size = height * width;

    for (int i = 0; i < size; ++i) {
        int max_idx = 0;
        float max_val = output_data[i];
        for (int j = 1; j < num_classes; ++j) {
            float val = output_data[j * size + i];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        label_map.data[i] = static_cast<uchar>(max_idx);
    }
    return label_map;
}

// label_map: getLabelMap() 的输出
// label_values: 要合并的类别索引列表（如 {1, 13} 表示皮肤和头发）
// 返回值: 255 表示属于这些类别的像素，0 表示不是
Mat FaceParsing::getCombinedMask(const Mat& label_map, const std::vector<int>& label_values) {
    CV_Assert(label_map.type() == CV_8UC1);

    Mat combined_mask = Mat::zeros(label_map.size(), CV_8UC1);

    for (int label : label_values) {
        combined_mask |= (label_map == label);
    }

    combined_mask.convertTo(combined_mask, CV_8UC1, 255);
    return combined_mask;
}

void FaceParsing::inferImage(Mat& src, Mat& dst) {
    this->preprocess(src);
    std::array<int64_t,4> input_shape {1,3,this->inpHeight, this->inpWidth};

    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info_handler, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    vector<Value> ort_outputs = this -> forward(input_tensor_);

    // 读取输出 & 校验 [1,19,512,512]
    float* out_data = ort_outputs.front().GetTensorMutableData<float>();
    auto info = ort_outputs.front().GetTensorTypeAndShapeInfo();
    auto dims = info.GetShape();
    assert(dims.size()==4 && dims[1]==19 && dims[2]==512 && dims[3]==512);

    // Argmax 得到 class_idx (CV_8UC1)
    int H=512, W=512, C=19;
    cv::Mat class_idx = getLabelMap(out_data,C,H,W);

    // 提取“皮肤”区域作为人脸轮廓基础 (类别ID=1)
    cv::Mat mask = (class_idx==1);

    // 形态学闭运算：填补小孔、平滑边界
    int ksize = 25;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 提取并绘制外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contour_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(contour_mask, contours, -1, cv::Scalar(255), cv::FILLED);

    dst = contour_mask;
}