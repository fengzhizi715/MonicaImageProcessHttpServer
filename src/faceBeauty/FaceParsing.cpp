//
// Created by Tony on 2025/5/20.
//
#include "../../include/faceBeauty/FaceParsing.h"

FaceParsing::FaceParsing(std::string modelPath, const char* logId, const char* provider): OnnxRuntimeBase(modelPath, logId, provider)
{
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->outHeight = output_node_dims[0][2];
    this->outWidth = output_node_dims[0][3];
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

void FaceParsing::inferImage(Mat& src, Mat& dst) {
    this->preprocess(src);
    std::array<int64_t,4> input_shape {1,3,this->inpHeight, this->inpWidth};

    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    vector<Value> ort_outputs = this -> forward(input_tensor_);

    // 读取输出 & 校验 [1,19,512,512] :contentReference[oaicite:5]{index=5}
    float* out_data = ort_outputs.front().GetTensorMutableData<float>();
    auto info = ort_outputs.front().GetTensorTypeAndShapeInfo();
    auto dims = info.GetShape();
    assert(dims.size()==4 && dims[1]==19 && dims[2]==512 && dims[3]==512);

    // Argmax 得到 class_idx (CV_8UC1)
    int H=512, W=512, C=19;
    cv::Mat class_idx(H,W,CV_8UC1);
    for(int h=0;h<H;++h){
        for(int w=0;w<W;++w){
            int bestc=0;
            float bestv=out_data[h*W + w];
            for(int c=1;c<C;++c){
                float v = out_data[c*H*W + h*W + w];
                if(v>bestv){ bestv=v; bestc=c; }
            }
            class_idx.at<uchar>(h,w) = static_cast<uchar>(bestc);
        }
    }

    // 提取“皮肤”区域作为人脸轮廓基础 (类别ID=1) :contentReference[oaicite:6]{index=6}
    cv::Mat mask = (class_idx==1);

    // 形态学闭运算：填补小孔、平滑边界:contentReference[oaicite:9]{index=9}
    int ksize = 25;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 提取并绘制外轮廓:contentReference[oaicite:10]{index=10}
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contour_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(contour_mask, contours, -1, cv::Scalar(255), cv::FILLED);

    dst = contour_mask;
}