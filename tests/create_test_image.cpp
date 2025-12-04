/**
 * Create a simple test image for the image processing server tests
 * Uses OpenCV to generate a test image
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Create a 256x256 RGB image with a blue background
    cv::Mat img(256, 256, CV_8UC3, cv::Scalar(73, 109, 137));
    
    // Draw a yellow rectangle
    cv::rectangle(img, cv::Point(50, 50), cv::Point(200, 150), cv::Scalar(0, 255, 255), -1);
    
    // Draw a red ellipse
    cv::ellipse(img, cv::Point(150, 150), cv::Size(50, 50), 0, 0, 360, cv::Scalar(0, 0, 255), -1);
    
    // Draw a green line
    cv::line(img, cv::Point(0, 0), cv::Point(256, 256), cv::Scalar(0, 255, 0), 5);
    
    // Save the image
    bool success = cv::imwrite("test_image.jpg", img);
    
    if (success) {
        std::cout << "Created test_image.jpg" << std::endl;
        return 0;
    } else {
        std::cerr << "Failed to create test image" << std::endl;
        return 1;
    }
}