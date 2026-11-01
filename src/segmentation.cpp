#include "rtvm/segmentation.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

namespace rtvm {

std::vector<float> toModelInput(const cv::Mat& bgr) {
    if (bgr.type() != CV_8UC3) {
        throw std::invalid_argument("toModelInput: expected an 8-bit BGR image");
    }

    std::vector<float> result(3 * bgr.rows * bgr.cols);
    for (int row = 0; row < bgr.rows; ++row) {
        for (int col = 0; col < bgr.cols; ++col) {
            cv::Vec3b pixel = bgr.at<cv::Vec3b>(row, col);

            const float R = (pixel[2] / 255.0f - 0.485f) / 0.229f;
            const float G = (pixel[1] / 255.0f - 0.456f) / 0.224f;
            const float B = (pixel[0] / 255.0f - 0.406f) / 0.225f;

            const int pixel_index = row * bgr.cols + col;
            result[0 * bgr.rows * bgr.cols + pixel_index] = R;
            result[1 * bgr.rows * bgr.cols + pixel_index] = G;
            result[2 * bgr.rows * bgr.cols + pixel_index] = B;
        }
    }
    return result;
}

cv::Mat personMaskFromLogits(const float* logits, int classes, int height, int width,
                             int person_class) {
    cv::Mat mask(height, width, CV_8U, cv::Scalar(0));      // all 0 to start

    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            int best_class = 0;
            float best_score = logits[row * width + col];

            for (int c = 1; c < classes; ++c) {
                float score = logits[c * height * width + row * width + col];
                if (score > best_score) {
                    best_class = c;
                    best_score = score;
                }
            }

            if (best_class == person_class) {
                mask.at<uchar>(row, col) = 255;
            }
        }
    }
    return mask;
}

// The ONNX Runtime objects. The session is the loaded model, so it lives as long as the segmenter.
struct PersonSegmenter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rtvm"};   // ONNX Runtime's global state; first, the session needs it
    Ort::Session session{nullptr};                     // empty until the constructor loads the model

    Impl(const std::string& model_path, bool use_coreml) {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("PersonSegmenter: model not found: " + model_path);
        }
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (use_coreml) {
            options.AppendExecutionProvider("CoreML", {});   // Core ML where it can, CPU for the rest
        }
        // filesystem::path: Windows wants a wide-character path here.
        session = Ort::Session(env, std::filesystem::path(model_path).c_str(), options);
    }
};

PersonSegmenter::PersonSegmenter(const std::string& model_path, bool use_coreml)
    : impl_(std::make_unique<Impl>(model_path, use_coreml)) {}

PersonSegmenter::~PersonSegmenter() = default;

cv::Mat PersonSegmenter::segment(const cv::Mat& bgr) {
    // The model was exported for 640x480.
    constexpr int kWidth = 640;
    constexpr int kHeight = 480;
    cv::Mat input = bgr;
    if (bgr.cols != kWidth || bgr.rows != kHeight) {
        cv::resize(bgr, input, cv::Size(kWidth, kHeight));
    }

    // Wrap our values in a tensor (no copy: it points into `values`, which must stay alive).
    std::vector<float> values = toModelInput(input);
    const std::array<std::int64_t, 4> shape{1, 3, kHeight, kWidth};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, values.data(), values.size(),
                                                              shape.data(), shape.size());

    // Run the model. The names are the ones given in scripts/export_segmentation.py.
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    std::vector<Ort::Value> outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names,
                                                         &tensor, 1, output_names, 1);

    // Output shape: 1 x classes x height x width.
    const std::vector<std::int64_t> out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    cv::Mat mask = personMaskFromLogits(outputs[0].GetTensorData<float>(),
                                        static_cast<int>(out_shape[1]),
                                        static_cast<int>(out_shape[2]),
                                        static_cast<int>(out_shape[3]));

    // Back to the input's size. Nearest neighbor: blending would create values between 0 and 255.
    if (mask.size() != bgr.size()) {
        cv::resize(mask, mask, bgr.size(), 0, 0, cv::INTER_NEAREST);
    }
    return mask;
}

}  // namespace rtvm