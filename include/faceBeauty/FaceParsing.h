//
// Created by Tony on 2025/5/20.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_FACEPARSING_H
#define MONICAIMAGEPROCESSHTTPSERVER_FACEPARSING_H

#include "../onnxruntime/OnnxRuntimeBase.h"

using namespace cv;
using namespace std;
using namespace Ort;

class FaceParsing: public OnnxRuntimeBase {
public:
    FaceParsing(std::string modelPath, const char* logId, const char* provider);

    /**
     * 通过推理获取人脸的 parsing
     * @param src
     * @param dst
     */
    void inferImage(Mat& src, Mat& dst);

    Mat getCombinedMask(const Mat& label_map, const std::vector<int>& label_values);

    void getSkinMask(Mat& parsing_result, Mat& dst);

private:
    void preprocess(Mat src);
    Mat getLabelMap(float* output_data, int num_classes, int height, int width);

    vector<float> input_image_;

    int inpWidth;
    int inpHeight;

    // 定义调色板（BGR）
    std::vector<cv::Vec3b> palette = {
            {  0,   0,   0}, {128,   0,   0}, {  0, 128,   0}, {128, 128,   0},
            {  0,   0, 128}, {128,   0, 128}, {  0, 128, 128}, {128, 128, 128},
            { 64,   0,   0}, {192,   0,   0}, { 64, 128,   0}, {192, 128,   0},
            { 64,   0, 128}, {192,   0, 128}, { 64, 128, 128}, {192, 128, 128},
            {  0,  64,   0}, {128,  64,   0}, {  0, 192,   0}
    };
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_FACEPARSING_H
