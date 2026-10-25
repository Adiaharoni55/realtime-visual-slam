#include "rtvm/features.hpp"

#include <opencv2/features2d.hpp>

namespace rtvm {

FrameFeatures detectOrb(const cv::Mat& gray, int max_features) {
    FrameFeatures features;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(max_features);
    orb->detectAndCompute(gray, cv::noArray(), features.keypoints, features.descriptors);
    return features;
}

std::vector<cv::DMatch> matchFeatures(const FrameFeatures& a, const FrameFeatures& b,
                                      double ratio) {
    std::vector<cv::DMatch> good;
    if (a.descriptors.empty() || b.descriptors.empty()) {
        return good;   // nothing to match
    }

    // For each descriptor in a, find its 2 closest descriptors in b.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher.knnMatch(a.descriptors, b.descriptors, candidates, 2);

    // Ratio test: keep a match only if it is clearly better than the runner-up.
    for (const std::vector<cv::DMatch>& pair : candidates) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

std::vector<cv::Mat> descriptorRows(const cv::Mat& descriptors) {
    std::vector<cv::Mat> rows;
    rows.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i) {
        rows.push_back(descriptors.row(i));
    }
    return rows;
}


}  // namespace rtvm