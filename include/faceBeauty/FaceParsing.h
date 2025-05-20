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

    void inferImage(Mat& src, Mat& dst);

private:
    void preprocess(Mat src);
    vector<float> input_image_;

    int inpWidth;
    int inpHeight;
    int outWidth;
    int outHeight;
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_FACEPARSING_H
