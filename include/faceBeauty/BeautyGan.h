//
// Created by Tony on 2025/5/21.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_BEAUTYGAN_H
#define MONICAIMAGEPROCESSHTTPSERVER_BEAUTYGAN_H

#include "../onnxruntime/OnnxRuntimeBase.h"

using namespace cv;
using namespace std;
using namespace Ort;

class BeautyGan: public OnnxRuntimeBase {
public:
    BeautyGan(std::string modelPath, const char* logId, const char* provider);

    void inferImage(Mat& src, Mat makeup, Mat& dst);

private:
    void preprocess1(Mat image, Size& origin_size);
    void preprocess2(Mat image, Size& origin_size);
    Mat postprocess(float* output_data);

    vector<float> input_image_1;
    vector<float> input_image_2;

    int inpWidth;
    int inpHeight;
    int outWidth;
    int outHeight;
};



#endif //MONICAIMAGEPROCESSHTTPSERVER_BEAUTYGAN_H
