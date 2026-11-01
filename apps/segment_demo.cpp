#include <iostream>
#include <chrono>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rtvm/segmentation.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image> [--coreml]\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read " << argv[1] << '\n';
        return 1;
    }
    std::cout << "Image: " << image.cols << " x " << image.rows << '\n';

    const bool use_coreml = argc > 2 && std::string(argv[2]) == "--coreml";
    rtvm::PersonSegmenter segmenter("models/deeplabv3_mobilenet.onnx", use_coreml);

    segmenter.segment(image);   // the first run is slower (setup), so it isn't timed

    const int runs = 20;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        segmenter.segment(image);
    }
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << (use_coreml ? "Core ML: " : "CPU: ") << ms / runs << " ms per image\n";

    // Save the image with people tinted red, to check the mask by eye.
    const cv::Mat mask = segmenter.segment(image);
    cv::Mat overlay = image.clone();
    overlay.setTo(cv::Scalar(0, 0, 255), mask);                // red (OpenCV order: B, G, R)
    cv::addWeighted(image, 0.5, overlay, 0.5, 0.0, overlay);   // 50% image, 50% red
    cv::imwrite("output/person_mask_cpp.png", overlay);
    std::cout << "Saved output/person_mask_cpp.png\n";


    return 0;
}

