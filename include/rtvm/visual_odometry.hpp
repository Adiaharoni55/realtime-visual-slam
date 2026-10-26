#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rtvm/camera.hpp"
#include "rtvm/features.hpp"
#include "rtvm/loop_detector.hpp"
#include "rtvm/pose.hpp"
#include "rtvm/pose_graph.hpp"

namespace rtvm {

// Thresholds that decide when the current frame becomes a new keyframe.
struct VoParams {
    double max_translation = 0.05;                  // meters moved since the keyframe
    double max_rotation_deg = 5.0;                  // degrees rotated since the keyframe
    int min_inliers = 100;                          // fewer PnP inliers than this -> new keyframe
};

// What happened inside track() for the last frame.
struct TrackStats {
    int matches = 0;
    int inliers = 0;
    bool success = false;
    bool new_keyframe = false;

    // Only filled for frames that become keyframes
    int loop_candidate = -1;                        // id of the best older keyframe, -1 if none
    double loop_score = 0.0;                        // its BoW similarity score
};

// Frame-to-keyframe visual odometry: estimates the camera pose of each new frame
// by matching it against the current keyframe. A new keyframe is chosen when the
// camera moved or rotated enough, or when there are too few inliers.
// Every keyframe is also stored in a pose graph for later optimization.
class VisualOdometry {
public:
    explicit VisualOdometry(PinholeCamera camera, VoParams params = {});

    // Process the next frame (grayscale image + 16-bit TUM depth image).
    // Returns the camera pose of this frame, relative to the first frame.
    Pose track(const cv::Mat& gray, const cv::Mat& depth);

    // Turns on loop-closure candidate search and loads the vocabulary.
    // Throws if the file doesn't exist.
    void enableLoopDetection(const std::string& vocabulary_path, LoopParams params = {});

    // One pose per tracked frame, rebuilt from the keyframe poses in the graph.
    // Equal to the poses track() returned until the graph is optimized.
    std::vector<Pose> trajectory() const;

    // Getters
    int failures() const { return failures_; }
    int keyframes() const { return graph_.numNodes(); }
    const TrackStats& lastStats() const { return last_stats_; }
    const PoseGraph& graph() const { return graph_; }

private:
    // Where a frame was tracked from: its keyframe, and its pose relative to that keyframe.
    struct FrameAnchor {
        int keyframe_id = 0;
        Pose relative = Pose::Identity();
    };

    // Configuration
    PinholeCamera camera_;
    VoParams params_;

    // Current state
    Pose pose_ = Pose::Identity();                  // current camera pose (first frame = origin)
    int failures_ = 0;

    // Keyframes
    PoseGraph graph_;                               // all keyframes (nodes) and the motions between them (edges)
    std::unique_ptr<LoopDetector> loop_detector_;   // empty = loop detection off
    Pose keyframe_pose_ = Pose::Identity();         // world pose of the current keyframe
    bool has_keyframe_ = false;                     // false until the first frame has been seen
    FrameFeatures keyframe_features_;
    cv::Mat keyframe_depth_;
    int keyframe_id_ = 0;                           // the graph id of the current keyframe

    // Per-frame records
    TrackStats last_stats_;
    std::vector<FrameAnchor> anchors_;

    // Makes the given frame the new keyframe and adds it to the graph.
    void setKeyframe(FrameFeatures&& features, const cv::Mat& depth);
};

}  // namespace rtvm
