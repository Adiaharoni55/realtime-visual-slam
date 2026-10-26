#include "rtvm/visual_odometry.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtvm/motion.hpp"

namespace rtvm {

VisualOdometry::VisualOdometry(PinholeCamera camera, VoParams params)
    : camera_(std::move(camera)), params_(params) {}

void VisualOdometry::enableLoopDetection(const std::string& vocabulary_path, LoopParams params) {
    loop_detector_ = std::make_unique<LoopDetector>(vocabulary_path, params);
}

std::vector<Pose> VisualOdometry::trajectory() const {
    std::vector<Pose> poses;
    poses.reserve(anchors_.size());
    for (const FrameAnchor& anchor : anchors_) {
        const Pose frame_pose = graph_.pose(anchor.keyframe_id) * anchor.relative;
        poses.push_back(frame_pose);
    }
    return poses;
}

void VisualOdometry::setKeyframe(FrameFeatures&& features, const cv::Mat& depth) {
    last_stats_.new_keyframe = true;   // every path that creates a keyframe goes through here

    // Add the new keyframe to the graph. The edge uses the OLD keyframe pose and id,
    // so this must happen before they are overwritten below.
    const int id = graph_.addNode(pose_);
    if (has_keyframe_) {
        graph_.addEdge(keyframe_id_, id, relativePose(keyframe_pose_, pose_));
    }

    // Ask for an older look-alike before adding this keyframe, so it can't find itself.
    if (loop_detector_) {
        std::optional<LoopCandidate> candidate = loop_detector_->query(features.descriptors);
        if (candidate) {
            last_stats_.loop_candidate = candidate->keyframe_id;
            last_stats_.loop_score     = candidate->score;
        }
        loop_detector_->add(features.descriptors);
        if (loop_detector_->size() != graph_.numNodes()) {
            throw std::logic_error("VisualOdometry: loop detector and graph ids out of sync");
        }
    }

    keyframe_id_ = id;
    keyframe_features_ = std::move(features);
    keyframe_depth_ = depth.clone();
    keyframe_pose_ = pose_;
    has_keyframe_ = true;
}

Pose VisualOdometry::track(const cv::Mat& gray, const cv::Mat& depth) {
    FrameFeatures current = detectOrb(gray);
    last_stats_ = TrackStats{};

    if (!has_keyframe_) {
        setKeyframe(std::move(current), depth);
        last_stats_.success = true;
    } else {
        std::vector<cv::DMatch> matches = matchFeatures(keyframe_features_, current);
        MotionEstimate est = estimateMotion(keyframe_features_, current, matches,
                                            keyframe_depth_, camera_);

        last_stats_.inliers = static_cast<int>(est.inliers.size());
        last_stats_.matches = static_cast<int>(matches.size());

        if (est.success) {
            pose_ = keyframe_pose_ * est.motion;
            last_stats_.success = true;
            // Decide whether this frame becomes the new keyframe
            const double movement = est.motion.translation().norm();
            const double rotation_deg =
                (Eigen::AngleAxisd(est.motion.rotation()).angle() * 180.0)/EIGEN_PI;
            if (movement > params_.max_translation ||
                rotation_deg > params_.max_rotation_deg ||
                last_stats_.inliers < params_.min_inliers) {
                setKeyframe(std::move(current), depth);
            }
        } else {
            ++failures_;
            setKeyframe(std::move(current), depth);
        }
    }

    anchors_.push_back(FrameAnchor{keyframe_id_, relativePose(keyframe_pose_, pose_)});
    return pose_;
}

} // namespace rtvm
