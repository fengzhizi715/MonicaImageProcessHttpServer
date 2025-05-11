//
// Created by Tony on 2025/5/11.
//

#ifndef MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H
#define MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H

class CodeFormer
{
public:
    CodeFormer(string modelpath);
    Mat detect(Mat cv_image);
private:
    void preprocess(Mat srcimg);
    vector<float> input_image_;
    vector<double> input2_tensor;
    int inpWidth;
    int inpHeight;
    int outWidth;
    int outHeight;

    float min_max[2] = { -1,1 };

    //存储初始化获得的可执行网络
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CodeFormer");
    Ort::Session *ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

#endif //MONICAIMAGEPROCESSHTTPSERVER_CODEFORMER_H
