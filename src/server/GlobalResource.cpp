//
// Created by Tony on 2025/3/25.
//

#include <vector>
#include <string>
#include <iostream>
#include "../../include/server/GlobalResource.h"
#include "../../include/onnxruntime/Constants.h"
#include "../utils/aixlog.hpp"
#include "../../include/utils/Timer.h"

GlobalResource::GlobalResource(string modelPath): modelPath(modelPath) {

    string sketchDrawingMode = modelPath + "/opensketch_style_512x512.onnx";
    const string& onnx_logid = "Sketch Drawing";

#ifdef USE_GPU
    const string& onnx_provider = OnnxProviders::CUDA;
#else
    const string& onnx_provider = OnnxProviders::CPU;
#endif

    const char* provider = onnx_provider.c_str();
    sketchDrawing = std::make_unique<SketchDrawing>(sketchDrawingMode, onnx_logid.c_str(), provider);

    string faceProto = modelPath + "/opencv_face_detector.pbtxt";
    string faceModel = modelPath + "/opencv_face_detector_uint8.pb";
    string ageProto  = modelPath + "/age_deploy.prototxt";
    string ageModel  = modelPath + "/age_net.caffemodel";
    string genderProto = modelPath + "/gender_deploy.prototxt";
    string genderModel = modelPath + "/gender_net.caffemodel";
    faceDetect = std::make_unique<FaceDetect>(faceProto,faceModel,ageProto,ageModel,genderProto,genderModel);

    string yolov8FaceModelPath = modelPath + "/yoloface_8n.onnx";
    string face68LandmarksModePath = modelPath + "/2dfan4.onnx";
    string faceEmbeddingModePath = modelPath + "/arcface_w600k_r50.onnx";
    string faceSwapModePath = modelPath + "/inswapper_128.onnx";
    string faceSwapModePath2 = modelPath + "/model_matrix.bin";
    string faceEnhanceModePath = modelPath + "/gfpgan_1.4.onnx";

    const std::string& yolov8FaceLogId = "yolov8Face";
    const std::string& face68LandmarksLogId = "face68Landmarks";
    const std::string& faceEmbeddingLogId = "faceEmbedding";
    const std::string& faceSwapLogId = "faceSwap";
    const std::string& faceEnhanceLogId = "faceEnhance";

    yolov8Face      = std::make_unique<Yolov8Face>(yolov8FaceModelPath, yolov8FaceLogId.c_str(), provider);
    face68Landmarks = std::make_unique<Face68Landmarks>(face68LandmarksModePath, face68LandmarksLogId.c_str(), provider);
    faceEmbedding   = std::make_unique<FaceEmbedding>(faceEmbeddingModePath, faceEmbeddingLogId.c_str(), provider);
    faceSwap        = std::make_unique<FaceSwap>(faceSwapModePath, faceSwapModePath2, faceSwapLogId.c_str(), provider);
    faceEnhance     = std::make_unique<FaceEnhance>(faceEnhanceModePath, faceEnhanceLogId.c_str(), provider);


    string animeGANHayaoModePath = modelPath + "/AnimeGANv3_Hayao_36.onnx";
    string animeGANJPFaceModePath = modelPath + "/AnimeGANv3_JP_face_v1.0.onnx";
    string animeGANPortraitSketchModePath = modelPath + "/AnimeGANv3_PortraitSketch_25.onnx";
    string animeGANShinkaiModePath = modelPath + "/AnimeGANv3_Shinkai_37.onnx";
    string animeGANTinyCuteModePath = modelPath + "/AnimeGANv3_tiny_Cute.onnx";

    const std::string& animeGANHayaoLogId = "animeGANHayao";
    const std::string& animeGANJPFaceLogId = "animeGANJPFace";
    const std::string& animeGANPortraitSketchLogId = "animeGANPortraitSketch";
    const std::string& animeGANShinkaiLogId = "animeGANShinkai";
    const std::string& animeGANTinyCuteLogId = "animeGANTinyCute";

    animeGANHayao          = std::make_unique<AnimeGAN>(animeGANHayaoModePath, animeGANHayaoLogId.c_str(), provider);
    animeGANJPFace         = std::make_unique<AnimeGAN>(animeGANJPFaceModePath, animeGANJPFaceLogId.c_str(), provider);
    animeGANPortraitSketch = std::make_unique<AnimeGAN>(animeGANPortraitSketchModePath, animeGANPortraitSketchLogId.c_str(), provider);
    animeGANShinkai        = std::make_unique<AnimeGAN>(animeGANShinkaiModePath, animeGANShinkaiLogId.c_str(), provider);
    animeGANTinyCute       = std::make_unique<AnimeGAN>(animeGANTinyCuteModePath, animeGANTinyCuteLogId.c_str(), provider);


    string beautyGanModePath = modelPath + "/beautygan.onnx";
    string codeFormerModePath = modelPath + "/codeformer.onnx";
    string faceParsingModePath = modelPath + "/face_parsing_resnet34.onnx";

    const std::string& beautyGanLogId = "beautyGan";
    const std::string& codeFormerLogId = "codeFormer";
    const std::string& faceParsingLogId = "faceParsing";

    beautyGan      = std::make_unique<BeautyGan>(beautyGanModePath,beautyGanLogId.c_str(), provider);
    codeFormer     = std::make_unique<CodeFormer>(codeFormerModePath,codeFormerLogId.c_str(), provider);
    faceParsing    = std::make_unique<FaceParsing>(faceParsingModePath,faceParsingLogId.c_str(), provider);

    string modnetModePath = modelPath + "/modnet.onnx";

    const std::string& modnetLogId = "modnet";
    modnet         = std::make_unique<Modnet>(modnetModePath,modnetLogId.c_str(), provider);

    // 初始化资源，加载模型文件
    PLOG(L_INFO) << "GlobalResource initialized." << std::endl;
}

Mat GlobalResource::processSketchDrawing(Mat src) {
    PLOG(L_INFO) << "process SketchDrawing..." << std::endl;

    Mat dst;
    sketchDrawing.get()->inferImage(src, dst);
    cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
    return dst;
}

Mat GlobalResource::processFaceDetect(Mat src) {
    PLOG(L_INFO) << "process FaceDetect..." << endl;

    Mat dst;
    faceDetect.get()->inferImage(src, dst);
    return dst;
}

Mat GlobalResource::processFaceLandMark(Mat src) {
    PLOG(L_INFO) << "process FaceLandMark..." << endl;

    Mat dst;

    cv::Scalar scalar(0, 0, 255);
    try {
        vector<Bbox> boxes;
        yolov8Face.get()->detect(src, boxes);
        dst = src.clone();
        for (auto box: boxes) {
           rectangle(dst, cv::Point(box.xmin,box.ymin), cv::Point(box.xmax,box.ymax), scalar, 4, 8, 0);

           vector<Point2f> face_landmark_5of68;
           face68Landmarks.get()->detect(src, box, face_landmark_5of68);
           for (auto point : face_landmark_5of68)
           {
               circle(dst, cv::Point(point.x, point.y), 4, scalar, -1);
           }
        }
    } catch(...) {
    }
    return dst;
}

Mat GlobalResource::processFaceSwap(Mat src, Mat target, bool status) {
    PLOG(L_INFO) << "process FaceSwap..." << endl;

    double processTime = 0.0;
    Timer processTimer = Timer(processTime, true);

    vector<Bbox> boxes;
    yolov8Face->detect(src, boxes);
    int position = 0; // 一张图片里可能有多个人脸，这里只考虑1个人脸的情况

    Bbox firstBox = boxes[position];

    vector<Point2f> face_landmark_5of68;
    face68Landmarks.get()->detect(src, boxes[position], face_landmark_5of68);
    vector<float> source_face_embedding = faceEmbedding.get()->detect(src, face_landmark_5of68);
    yolov8Face.get() -> detect(target, boxes);
    Mat dst = target.clone();

    if (!boxes.empty()) {
        if (status) {
            for (auto box: boxes) {
                vector<Point2f> target_landmark_5;
                face68Landmarks.get()->detect(dst, box, target_landmark_5);

                Mat swap = faceSwap.get()->process(dst, source_face_embedding, target_landmark_5);
                dst = faceEnhance.get()->process(swap, target_landmark_5);
            }
        } else {
            Bbox  box = boxes[0];
            vector<Point2f> target_landmark_5;
            face68Landmarks.get()->detect(dst, box, target_landmark_5);
            Mat swap = faceSwap.get()->process(dst, source_face_embedding, target_landmark_5);
            dst = faceEnhance.get()->process(swap, target_landmark_5);
        }
    }

    processTimer.stop();
    PLOG(L_INFO) << "GlobalResource::processFaceSwap function take " << (processTime * 1000.0) << "ms to complete the process" << endl;

    return dst;
}

Mat GlobalResource::processCartoon(Mat src, int type) {
    PLOG(L_INFO) << "process Cartoon..." << endl;

    double processTime = 0.0;
    Timer processTimer = Timer(processTime, true);

    Mat dst;
    switch(type) {
        case 1:
            animeGANHayao.get()->inferImage(src, dst);
            break;
        case 2:
            animeGANJPFace.get()->inferImage(src, dst);
            break;
        case 3:
            animeGANPortraitSketch.get()->inferImage(src, dst);
            break;
        case 4:
            animeGANShinkai.get()->inferImage(src, dst);
            break;
        case 5:
            animeGANTinyCute.get()->inferImage(src, dst);
            break;
        default:
            animeGANHayao.get()->inferImage(src, dst);
    }

    processTimer.stop();
    PLOG(L_INFO) << "GlobalResource::processCartoon function take " << (processTime * 1000.0) << "ms to complete the process" << endl;

    return dst;
}

// 美颜增强流水线: YOLOv8 -> Face Parsing -> BeautyGAN -> CodeFormer -> Mask 融合回原图
Mat GlobalResource::processBeauty(Mat src, Mat makeup) {
    PLOG(L_INFO) << "process Beauty..." << endl;

    // 人脸检测与裁剪
    Mat original_face;
    Bbox box;
    try {
        vector<Bbox> boxes;
        yolov8Face.get()->detect(src, boxes);
        box = boxes[0];

        original_face = src(Rect(cv::Point(box.xmin,box.ymin), cv::Point(box.xmax,box.ymax)));
    } catch(...) {
    }

    // 查找人脸的关键点
    vector<Point2f> face_landmark_5of68;
    face68Landmarks.get()->detect(src, box, face_landmark_5of68);

    // 人脸解析与获取掩码
    cv::Mat parsing_result, skin_mask;
    faceParsing.get()->inferImage(original_face, parsing_result);
    faceParsing.get()->getSkinMask(parsing_result, skin_mask);

    // 美颜模型处理
    Mat beautygan_crop;
    beautyGan.get()->inferImage(original_face, makeup, beautygan_crop);

    // 使用 CodeFormer 对人脸优化细节
    Mat codeformed_face; // CodeFormer 输出的人脸
    codeFormer.get()->inferImage(beautygan_crop, codeformed_face);
    resize(codeformed_face, codeformed_face, original_face.size());

    // 通过掩码将处理后人脸融回原图
    cv::resize(skin_mask, skin_mask, codeformed_face.size());
    Point center(original_face.cols/2, original_face.rows/2);
    cv::Mat blended_face;
    cv::seamlessClone(codeformed_face, original_face, skin_mask, center, blended_face, NORMAL_CLONE);
    blended_face.copyTo(src(Rect(cv::Point(box.xmin,box.ymin), cv::Point(box.xmax,box.ymax))));

    // 最后再用 GFPGAN 模型对人脸进行增强
    Mat result = faceEnhance.get()->process(src, face_landmark_5of68);
    return result;
}

Mat GlobalResource::processPersonBackground(Mat src, Mat background) {
    PLOG(L_INFO) << "process change person background..." << endl;

    double processTime = 0.0;
    Timer processTimer = Timer(processTime, true);

    Mat dst;
    modnet.get()->changeBackground(src,background,dst);

    processTimer.stop();
    PLOG(L_INFO) << "GlobalResource::processPersonBackground function take " << (processTime * 1000.0) << "ms to complete the process" << endl;

    return dst;
}

Mat GlobalResource::changeHairColor(Mat src, int target_hue, float saturation_scale) {
    PLOG(L_INFO) << "process change hair color..." << endl;

    double processTime = 0.0;
    Timer processTimer = Timer(processTime, true);

    Mat mask;
    modnet.get()->inferImage(src, mask);

    cv::Rect face_roi = getSmartFaceROIFromAlpha(mask);
    cv::Mat face_crop = src(face_roi).clone();

    // 人脸解析与获取掩码
    cv::Mat class_idx;
    faceParsing.get()->inferImage(face_crop, class_idx);

    // 提取头发区域的 mask
    cv::Mat hair_mask_small = (class_idx == 17);
    hair_mask_small.convertTo(hair_mask_small, CV_8UC1, 255);

    // Resize 回 ROI 尺寸
    cv::Mat hair_mask_roi;
    cv::resize(hair_mask_small, hair_mask_roi, face_roi.size(), 0, 0, cv::INTER_NEAREST);

    // 映射回原图尺寸
    cv::Mat hair_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    hair_mask_roi.copyTo(hair_mask(face_roi));

    cv::Mat recolored = changeHairColor_HSV(src, hair_mask, target_hue, saturation_scale);

    processTimer.stop();
    PLOG(L_INFO) << "GlobalResource::changeHairColor function take " << (processTime * 1000.0) << "ms to complete the process" << endl;

    return recolored;
}