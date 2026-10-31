#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtvm/camera.hpp"
#include "rtvm/features.hpp"
#include "rtvm/loop_detector.hpp"
#include "rtvm/pose.hpp"
#include "rtvm/pose_graph.hpp"

namespace rtvm {

// Tuning parameters for tracking, keyframe selection, loop verification and the sanity check.
struct VoParams {
    double max_translation = 0.05;                  // meters moved since the keyframe
    double max_rotation_deg = 5.0;                  // degrees rotated since the keyframe
    int min_inliers = 100;                          // fewer PnP inliers than this -> new keyframe
    int min_loop_inliers = 50;                      // PnP inliers needed to accept a loop candidate
    int min_track_inliers = 20;                     // fewer PnP inliers than this: don't trust the estimate
    double max_frame_translation = 0.15;            // meters the camera may move between two frames
    bool local_ba = false;                          // local bundle adjustment after each new keyframe
    int ba_window = 10;                             // how many of the newest keyframes BA may move
    int ba_min_observations = 20;                   // a window keyframe with fewer observations in
                                                    // the problem is fixed, not free (else degenerate)
    double ba_depth_noise = 0.01;                   // depth sigma = this * z^2 (see BundleAdjuster)
};

// What happened inside track() for the last frame.
struct TrackStats {
    int matches = 0;
    int inliers = 0;
    bool success = false;
    bool rejected = false;                          // PnP succeeded, but the sanity check refused the estimate
    bool new_keyframe = false;

    // Only filled for frames that become keyframes
    int loop_candidate = -1;                        // id of the best older keyframe, -1 if none
    double loop_score = 0.0;                        // its BoW similarity score
    int loop_inliers = 0;                           // PnP inliers when verifying the candidate
    bool loop_accepted = false;                     // true if the loop edge was added
};

// Size and connectivity of the local map. A point seen by one keyframe only gives
// bundle adjustment nothing to work with, so the observations per point matter.
struct MapStats {
    int points = 0;
    int observations = 0;      // summed over all points
    int points_3plus = 0;      // points seen by at least 3 keyframes
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

    // Optimizes the pose graph (uses the loop edges). Chain edges are first re-measured from the
    // current poses, so poses refined by local BA aren't pulled back to the old PnP values.
    // Afterwards trajectory() returns the corrected poses. Call once, at the end.
    // Tracking state isn't updated.
    void optimize() {
        graph_.refreshChainEdges();
        graph_.optimize();
    }

    // Getters
    int failures() const { return failures_; }
    int loopEdges() const { return loop_edges_; }
    int baRuns() const { return ba_runs_; }              // how many local BAs were solved
    double baSeconds() const { return ba_seconds_; }     // their total time (s)
    int keyframes() const { return graph_.numNodes(); }
    const TrackStats& lastStats() const { return last_stats_; }
    const PoseGraph& graph() const { return graph_; }
    MapStats mapStats() const;


private:
    // Where a frame was tracked from: its keyframe, and its pose relative to that keyframe.
    struct FrameAnchor {
        int keyframe_id = 0;
        Pose relative = Pose::Identity();
    };

    // Everything stored per keyframe (index = graph id).
    struct Keyframe {
        FrameFeatures features;
        cv::Mat depth;
        std::vector<Eigen::Vector2d> undistorted;   // undistorted keypoint pixels (for BA)
        std::vector<int> map_point;                 // per keypoint: its map point id, or -1
    };

    // A 3D point and the keyframes that saw it.
    struct MapPoint {
        Eigen::Vector3d position;                          // world coordinates
        std::vector<std::pair<int, int>> observations;     // (keyframe id, keypoint index)
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
    std::vector<Keyframe> keyframes_;               // every keyframe's data, by graph id
    std::vector<MapPoint> map_points_;              // the local map, id = index
    int keyframe_id_ = 0;                           // the graph id of the current keyframe
    int loop_edges_ = 0;
    int ba_runs_ = 0;
    double ba_seconds_ = 0.0;

    // Per-frame records
    TrackStats last_stats_;
    std::vector<FrameAnchor> anchors_;

    // associations: (keypoint in the current keyframe, keypoint in the new one) for matches
    // that PnP accepted. They continue existing map points into the new keyframe.
    void setKeyframe(FrameFeatures&& features, const cv::Mat& depth,
                    const std::vector<std::pair<int, int>>& associations = {});

    // Bundle adjustment over the newest ba_window keyframes and the map points they see;
    // writes the refined poses and points back
    void localBundleAdjust();
};

}  // namespace rtvm
