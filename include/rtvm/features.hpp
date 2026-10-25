#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace rtvm {

// Keypoints of one image and their ORB descriptors (one row per keypoint).
struct FrameFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

// Detect up to max_features ORB keypoints in a grayscale image.
FrameFeatures detectOrb(const cv::Mat& gray, int max_features = 1000);

// Match features from frame a to frame b (Hamming distance + Lowe ratio test).
// Each returned match links a.keypoints[m.queryIdx] to b.keypoints[m.trainIdx].
std::vector<cv::DMatch> matchFeatures(const FrameFeatures& a, const FrameFeatures& b,
                                      double ratio = 0.75);

// Split an Nx32 descriptor matrix into N 1x32 rows (the format DBoW2 expects).
std::vector<cv::Mat> descriptorRows(const cv::Mat& descriptors);

}  // namespace rtvm