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
            const int c = candidate->keyframe_id;
            last_stats_.loop_candidate = c;
            last_stats_.loop_score     = candidate->score;
            std::vector<cv::DMatch> matches = matchFeatures(keyframe_features_[c], features);
            MotionEstimate est = estimateMotion(keyframe_features_[c], features, matches,
                                                keyframe_depths_[c], camera_);
            last_stats_.loop_inliers = static_cast<int>(est.inliers.size());
            // Verify geometrically: PnP from the old keyframe to this one. Only then trust it as a loop edge.
            if (est.success && last_stats_.loop_inliers >= params_.min_loop_inliers) {
                graph_.addEdge(c, id, est.motion);
                ++loop_edges_;
                last_stats_.loop_accepted = true;
            }
        }
        loop_detector_->add(features.descriptors);
    }

    keyframe_id_ = id;
    keyframe_features_.push_back(std::move(features));
    keyframe_depths_.push_back(depth.clone());
    keyframe_pose_ = pose_;
    has_keyframe_ = true;

    // Everything stored per keyframe must have one entry per graph node, or an id would
    // point to the wrong keyframe's data.
    const int n = graph_.numNodes();
    if (static_cast<int>(keyframe_features_.size()) != n ||
        static_cast<int>(keyframe_depths_.size()) != n ||
        (loop_detector_ && loop_detector_->size() != n)) {
        throw std::logic_error("VisualOdometry: keyframe data and graph ids out of sync");
    }
}

Pose VisualOdometry::track(const cv::Mat& gray, const cv::Mat& depth) {
    FrameFeatures current = detectOrb(gray);
    last_stats_ = TrackStats{};

    if (!has_keyframe_) {
        setKeyframe(std::move(current), depth);
        last_stats_.success = true;
    } else {
        // References into keyframe_features_/keyframe_depths_: only valid until the next
        // setKeyframe() (push_back may reallocate). Don't use them after it.
        const FrameFeatures& kf_features = keyframe_features_[keyframe_id_];
        const cv::Mat& kf_depth = keyframe_depths_[keyframe_id_];

        std::vector<cv::DMatch> matches = matchFeatures(kf_features, current);
        MotionEstimate est = estimateMotion(kf_features, current, matches,
                                            kf_depth, camera_);

        last_stats_.inliers = static_cast<int>(est.inliers.size());
        last_stats_.matches = static_cast<int>(matches.size());

        // Sanity check: PnP can "succeed" with a few wrong matches that happen to agree.
        // Accept only with enough inliers and a physically possible step since the last frame
        // (pose_ still holds the previous frame's pose here).
        bool accepted = false;
        Pose candidate = pose_;
        if (est.success) {
            candidate = keyframe_pose_ * est.motion;
            const double step = relativePose(pose_, candidate).translation().norm();
            accepted = last_stats_.inliers >= params_.min_track_inliers &&
                       step <= params_.max_frame_translation;
            last_stats_.rejected = !accepted;
        }

        if (accepted) {
            pose_ = candidate;
            last_stats_.success = true;
            // Decide whether this frame becomes the new keyframe
            const double movement = est.motion.translation().norm();
            const double rotation_deg =
                (Eigen::AngleAxisd(est.motion.rotation()).angle() * 180.0) / EIGEN_PI;
            if (movement > params_.max_translation ||
                rotation_deg > params_.max_rotation_deg ||
                last_stats_.inliers < params_.min_inliers) {
                setKeyframe(std::move(current), depth);
            }
        } else {
            // PnP failed or the estimate is implausible: keep the last pose and restart
            // from this frame as a new keyframe.
            ++failures_;
            setKeyframe(std::move(current), depth);
        }
    }

    anchors_.push_back(FrameAnchor{keyframe_id_, relativePose(keyframe_pose_, pose_)});
    return pose_;
}

} // namespace rtvm
