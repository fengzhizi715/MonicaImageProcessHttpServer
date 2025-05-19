//
// Created by Tony on 2025/5/19.
//
#include "../../include/faceBeauty/CodeFormer.h"

CodeFormer::CodeFormer(std::string modelPath, const char* logId, const char* provider): OnnxRuntimeBase(modelPath, logId, provider)
{
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->outHeight = output_node_dims[0][2];
    this->outWidth = output_node_dims[0][3];
    input2_tensor.push_back(0.5);
}

void CodeFormer::preprocess(Mat src)
{
    Mat dst;
    cvtColor(src, dst, COLOR_BGR2RGB);
    resize(dst, dst, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
    this->input_image_.resize(this->inpWidth * this->inpHeight * dst.channels());
    int k = 0;
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < this->inpHeight; i++)
        {
            for (int j = 0; j < this->inpWidth; j++)
            {
                float pix = dst.ptr<uchar>(i)[j * 3 + c];
                this->input_image_[k] = (pix / 255.0 - 0.5) / 0.5;
                k++;
            }
        }
    }
}

void CodeFormer::inferImage(Mat& src, Mat& dst)
{
    int im_h = src.rows;
    int im_w = src.cols;
    this->preprocess(src);
    array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
    vector<int64_t> input2_shape_ = { 1 };

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    vector<Value> ort_inputs;
    ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()));
    ort_inputs.push_back(Value::CreateTensor<double>(allocator_info, input2_tensor.data(), input2_tensor.size(), input2_shape_.data(), input2_shape_.size()));
    vector<Value> ort_outputs = ort_session.Run(RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names.data(), output_names.size());

    ////post_process
    float* pred = ort_outputs[0].GetTensorMutableData<float>();
    //////Mat mask(outHeight, outWidth, CV_32FC3, pred); /////经过试验,直接这样赋值,是不行的
    const unsigned int channel_step = outHeight * outWidth;
    vector<Mat> channel_mats;
    Mat rmat(outHeight, outWidth, CV_32FC1, pred); // R
    Mat gmat(outHeight, outWidth, CV_32FC1, pred + channel_step); // G
    Mat bmat(outHeight, outWidth, CV_32FC1, pred + 2 * channel_step); // B
    channel_mats.push_back(rmat);
    channel_mats.push_back(gmat);
    channel_mats.push_back(bmat);

    merge(channel_mats, dst); // CV_32FC3 allocated

    ///不用for循环遍历Mat里的每个像素值,实现numpy.clip函数
    dst.setTo(this->min_max[0], dst < this->min_max[0]);
    dst.setTo(this->min_max[1], dst > this->min_max[1]);   ////也可以用threshold函数,阈值类型THRESH_TOZERO_INV

    dst = (dst - this->min_max[0]) / (this->min_max[1] - this->min_max[0]);
    dst *= 255.0;
    dst.convertTo(dst, CV_8UC3);
    cvtColor(dst, dst, COLOR_BGR2RGB);
}