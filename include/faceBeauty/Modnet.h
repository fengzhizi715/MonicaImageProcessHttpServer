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

    void inferImage(const Mat& src, Mat& mask);

    /**
     * 替换人物的背景
     * @param src
     * @param background
     * @param dst
     */
    void changeBackground(const Mat& src, const Mat& background, Mat& dst);

private:
    void preprocess(const cv::Mat& image);
    Mat postprocess(float* output_data, int width, int height);
    vector<float> input_image_;

    int inpWidth;
    int inpHeight;
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_MODNET_H
