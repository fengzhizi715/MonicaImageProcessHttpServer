//
// Created by Tony on 2025/3/25.
//

#ifndef MONICAIMAGEPROCESS_GLOBALRESOURCE_H
#define MONICAIMAGEPROCESS_GLOBALRESOURCE_H

#include <memory>
#include <mutex>
#include "../../include/cartoon/AnimeGAN.h"
#include "../../include/sketchDrawing/SketchDrawing.h"
#include "../../include/faceDetect/FaceDetect.h"
#include "../../include/faceSwap/Face68Landmarks.h"
#include "../../include/faceSwap/FaceEmbedding.h"
#include "../../include/faceSwap/FaceSwap.h"
#include "../../include/faceSwap/FaceEnhance.h"
#include "../../include/faceSwap/Yolov8Face.h"
#include "../../include/faceBeauty/BeautyGan.h"
#include "../../include/faceBeauty/CodeFormer.h"
#include "../../include/faceBeauty/FaceParsing.h"
#include "../../include/faceBeauty/Modnet.h"

class GlobalResource {
public:
    GlobalResource(std::string modelPath);
    cv::Mat processSketchDrawing(cv::Mat src);
    cv::Mat processFaceDetect(cv::Mat src);
    cv::Mat processFaceLandMark(cv::Mat src);
    cv::Mat processFaceSwap(cv::Mat src, cv::Mat target, bool status);
    cv::Mat processCartoon(cv::Mat src, int type);
    cv::Mat processBeauty(cv::Mat src, cv::Mat makeup);
    cv::Mat processPersonBackground(cv::Mat src, cv::Mat background);
    cv::Mat changeHairColor(cv::Mat src, int target_hue, float saturation_scale);

private:
    std::string modelPath;

    std::unique_ptr<SketchDrawing>   sketchDrawing;
    std::unique_ptr<FaceDetect>      faceDetect;
    std::unique_ptr<Yolov8Face>      yolov8Face;
    std::unique_ptr<Face68Landmarks> face68Landmarks;
    std::unique_ptr<FaceEmbedding>   faceEmbedding;
    std::unique_ptr<FaceSwap>        faceSwap;
    std::unique_ptr<FaceEnhance>     faceEnhance;

    std::unique_ptr<AnimeGAN>        animeGANHayao;
    std::unique_ptr<AnimeGAN>        animeGANJPFace;
    std::unique_ptr<AnimeGAN>        animeGANPortraitSketch;
    std::unique_ptr<AnimeGAN>        animeGANShinkai;
    std::unique_ptr<AnimeGAN>        animeGANTinyCute;

    std::unique_ptr<BeautyGan>       beautyGan;
    std::unique_ptr<CodeFormer>      codeFormer;
    std::unique_ptr<FaceParsing>     faceParsing;

    std::unique_ptr<Modnet>          modnet;
    
    // Thread-safe protection for shared model access
    mutable std::mutex sketchDrawing_mutex_;
    mutable std::mutex faceDetect_mutex_;
    mutable std::mutex yolov8Face_mutex_;
    mutable std::mutex face68Landmarks_mutex_;
    mutable std::mutex faceEmbedding_mutex_;
    mutable std::mutex faceSwap_mutex_;
    mutable std::mutex faceEnhance_mutex_;
    mutable std::mutex animeGAN_mutex_;
    mutable std::mutex beautyGan_mutex_;
    mutable std::mutex codeFormer_mutex_;
    mutable std::mutex faceParsing_mutex_;
    mutable std::mutex modnet_mutex_;
};

#endif //MONICAIMAGEPROCESS_GLOBALRESOURCE_H
