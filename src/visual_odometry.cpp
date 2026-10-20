#include "rtvm/visual_odometry.hpp"

#include <utility>
#include <vector>
#include <cmath>

#include "rtvm/motion.hpp"

namespace rtvm {
VisualOdometry::VisualOdometry(PinholeCamera camera, VoParams params)
    : camera_(std::move(camera)), params_(params) {}

void VisualOdometry::setKeyframe(FrameFeatures&& features, const cv::Mat& depth){
    keyframe_features_ = std::move(features);
    keyframe_depth_ = depth.clone();
    keyframe_pose_ = pose_;
    has_keyframe_ = true;
    ++num_keyframes_;
}

Pose VisualOdometry::track(const cv::Mat& gray, const cv::Mat& depth){
    FrameFeatures current = detectOrb(gray);

    if(!has_keyframe_){
        setKeyframe(std::move(current), depth);
        return pose_;
    }

    std::vector<cv::DMatch> matches = matchFeatures(keyframe_features_, current);
    MotionEstimate est = estimateMotion(keyframe_features_, current, matches,
                                        keyframe_depth_, camera_);
    if (est.success){
        pose_ = keyframe_pose_ * est.motion;
    } else {
        ++failures_;
        setKeyframe(std::move(current), depth);
        return pose_;
    }
    // Decide whether this frame becomes the new keyframe
    double movement = est.motion.translation().norm();
    double rotation_deg = (Eigen::AngleAxisd(est.motion.rotation()).angle() * 180.0)/EIGEN_PI;
    size_t num_inliers = est.inliers.size();
    if (movement > params_.max_translation ||
        rotation_deg > params_.max_rotation_deg ||
        num_inliers < params_.min_inliers) {
        setKeyframe(std::move(current), depth);
    }

    return pose_;
}

} // namespace rtvm
