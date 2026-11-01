#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rtvm/segmentation.hpp"

// A 1x2 image: pixel 0 pure red, pixel 1 pure blue (OpenCV stores B, G, R).
// Output order is planar: R0 R1 G0 G1 B0 B1, each value normalized with ImageNet mean/std.
TEST(SegmentationTest, ToModelInputIsPlanarAndNormalized) {
    cv::Mat bgr(1, 2, CV_8UC3);
    bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);   // red
    bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 0, 0);   // blue

    const std::vector<float> input = rtvm::toModelInput(bgr);

    ASSERT_EQ(input.size(), 6u);
    EXPECT_NEAR(input[0], (1.0f - 0.485f) / 0.229f, 1e-5);   // R of pixel 0
    EXPECT_NEAR(input[1], (0.0f - 0.485f) / 0.229f, 1e-5);   // R of pixel 1
    EXPECT_NEAR(input[2], (0.0f - 0.456f) / 0.224f, 1e-5);   // G of pixel 0
    EXPECT_NEAR(input[3], (0.0f - 0.456f) / 0.224f, 1e-5);   // G of pixel 1
    EXPECT_NEAR(input[4], (0.0f - 0.406f) / 0.225f, 1e-5);   // B of pixel 0
    EXPECT_NEAR(input[5], (1.0f - 0.406f) / 0.225f, 1e-5);   // B of pixel 1
}

// Only 8-bit, 3-channel images are valid input.
TEST(SegmentationTest, ToModelInputRejectsWrongType) {
    EXPECT_THROW(rtvm::toModelInput(cv::Mat(2, 2, CV_8UC1)), std::invalid_argument);   // grayscale
    EXPECT_THROW(rtvm::toModelInput(cv::Mat()), std::invalid_argument);                // empty
}

// A 2x2 image with 16 classes: class 15 wins at (0,0) and (1,1), class 3 at (0,1),
// and at (1,0) all scores are equal, so class 0 (background) stays the winner.
TEST(SegmentationTest, PersonMaskFromLogitsPicksTheWinner) {
    const int classes = 16;
    const int height = 2;
    const int width = 2;
    std::vector<float> logits(classes * height * width, 0.0f);
    auto score = [&](int c, int row, int col) -> float& {
        return logits[c * height * width + row * width + col];
    };
    score(15, 0, 0) = 1.0f;
    score(15, 1, 1) = 1.0f;
    score(3, 0, 1) = 1.0f;

    const cv::Mat mask = rtvm::personMaskFromLogits(logits.data(), classes, height, width);

    ASSERT_EQ(mask.type(), CV_8U);
    ASSERT_EQ(mask.rows, height);
    ASSERT_EQ(mask.cols, width);
    EXPECT_EQ(mask.at<uchar>(0, 0), 255);
    EXPECT_EQ(mask.at<uchar>(0, 1), 0);
    EXPECT_EQ(mask.at<uchar>(1, 0), 0);
    EXPECT_EQ(mask.at<uchar>(1, 1), 255);
}

TEST(SegmentationTest, MissingModelThrows) {
    EXPECT_THROW(rtvm::PersonSegmenter("does/not/exist.onnx"), std::runtime_error);
}

// The real model on a real image. Same image as scripts/export_segmentation.py, which found
// 10.3% person pixels. CI has neither the 44 MB model nor the dataset, so it skips this test.
TEST(SegmentationTest, FindsPersonOnRealImage) {
    const std::string root = std::string(RTVM_CONFIG_DIR) + "/..";
    const std::string model = root + "/models/deeplabv3_mobilenet.onnx";
    const std::string image =
        root + "/data/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846321.637163.png";
    if (!std::filesystem::exists(model) || !std::filesystem::exists(image)) {
        GTEST_SKIP() << "model or image not found";
    }

    const cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());
    rtvm::PersonSegmenter segmenter(model);
    const cv::Mat mask = segmenter.segment(bgr);

    ASSERT_EQ(mask.size(), bgr.size());
    ASSERT_EQ(mask.type(), CV_8U);
    const double person_share = static_cast<double>(cv::countNonZero(mask)) / mask.total();
    EXPECT_NEAR(person_share, 0.103, 0.02);
}
