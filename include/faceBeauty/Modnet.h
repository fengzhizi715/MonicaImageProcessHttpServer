//
// Created by Tony on 2025/6/16.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_MODNET_H
#define MONICAIMAGEPROCESSHTTPSERVER_MODNET_H

#include "../onnxruntime/OnnxRuntimeBase.h"

using namespace cv;
using namespace std;
using namespace Ort;

class Modnet: public OnnxRuntimeBase {
public:
    Modnet(std::string modelPath, const char* logId, const char* provider);

    void inferImage(Mat& src, Mat& mask);

private:
    void preprocess(const cv::Mat& image);
    Mat postprocess(float* output_data, int width, int height);
    vector<float> input_image_;

    int inpWidth;
    int inpHeight;
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_MODNET_H
