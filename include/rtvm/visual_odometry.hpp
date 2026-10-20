#pragma once

#include "rtvm/camera.hpp"
#include "rtvm/features.hpp"
#include "rtvm/pose.hpp"

namespace rtvm {

// Thresholds that decide when the current frame becomes a new keyframe.
struct VoParams {
    double max_translation = 0.05;   // meters moved since the keyframe
    double max_rotation_deg = 5.0;   // degrees rotated since the keyframe
    int min_inliers = 100;           // fewer PnP inliers than this -> new keyframe
};

struct TrackStats {
    int matches = 0;
    int inliers = 0;
    bool success = false;
    bool new_keyframe = false;
};

// Frame-to-frame visual odometry: estimates the camera pose of each new frame
// by matching it against the last keyframe. 
// New keyframe is chosen when the camera moved or rotated enough, or when there are too few inliers.
class VisualOdometry {
public:
    explicit VisualOdometry(PinholeCamera camera, VoParams params = {});

    // Process the next frame (grayscale image + 16-bit TUM depth image).
    // Returns the camera pose of this frame, relative to the first frame.
    Pose track(const cv::Mat& gray, const cv::Mat& depth);

    // Gets 
    int failures() const { return failures_; }
    int keyframes() const { return num_keyframes_; }
    const TrackStats& lastStats() const { return last_stats_; }

private:
    // parameters
    PinholeCamera camera_;
    VoParams params_;

    // Current State
    Pose pose_ = Pose::Identity();          // current camera pose (first frame = origin)
    int failures_ = 0;

    // Keyframe
    Pose keyframe_pose_ = Pose::Identity(); // world pose of the current keyframe
    bool has_keyframe_ = false;             // false until the first frame has been seen
    FrameFeatures keyframe_features_;
    cv::Mat keyframe_depth_;
    int num_keyframes_ = 0;

    // Tracking the status
    TrackStats last_stats_;

    // Function to set the keyframe when needed
    void setKeyframe(FrameFeatures&& features, const cv::Mat& depth);
};

}  // namespace rtvm
