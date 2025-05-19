//
// Created by Tony on 2025/5/11.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H
#define MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H

#include "../onnxruntime/OnnxRuntimeBase.h"

using namespace cv;
using namespace std;
using namespace Ort;

class CodeFormer: public OnnxRuntimeBase {
public:
    CodeFormer(std::string modelPath, const char* logId, const char* provider);

    void inferImage(Mat& src, Mat& dst);

private:
    void preprocess(Mat src);
    vector<float> input_image_;
    vector<double> input2_tensor;
    int inpWidth;
    int inpHeight;
    int outWidth;
    int outHeight;

    float min_max[2] = { -1,1 };
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H
