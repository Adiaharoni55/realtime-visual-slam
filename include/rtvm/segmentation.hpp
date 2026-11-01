#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rtvm {

// Prepare a BGR image for the model: BGR -> RGB, scaled to 0-1, ImageNet normalization.
// Returns planar CHW values: all red, then all green, then all blue.
std::vector<float> toModelInput(const cv::Mat& bgr);

// Turn the model's raw output (classes x height x width) into a mask:
// CV_8U, 255 where person_class has the highest score, 0 elsewhere.
cv::Mat personMaskFromLogits(const float* logits, int classes, int height, int width,
                             int person_class = 15);

// Finds people in an image with a segmentation network (DeepLabV3 + MobileNetV3, ONNX).
class PersonSegmenter {
public:
    // Loads the model. use_coreml: run on Apple's Core ML. Throws std::runtime_error if the
    // file doesn't exist.
    explicit PersonSegmenter(const std::string& model_path, bool use_coreml = false);
    ~PersonSegmenter();

    // Returns a CV_8U mask of the image's size: 255 on people, 0 elsewhere.
    cv::Mat segment(const cv::Mat& bgr);

private:
    struct Impl;                    // holds the ONNX Runtime objects, defined in the .cpp
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtvm
