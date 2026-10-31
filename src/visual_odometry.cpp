#include "rtvm/visual_odometry.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>

#include "rtvm/motion.hpp"
#include "rtvm/bundle_adjustment.hpp"

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

void VisualOdometry::setKeyframe(FrameFeatures&& features, const cv::Mat& depth,
                                 const std::vector<std::pair<int, int>>& associations) {
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
            std::vector<cv::DMatch> matches = matchFeatures(keyframes_[c].features, features);
            MotionEstimate est = estimateMotion(keyframes_[c].features, features, matches,
                                                keyframes_[c].depth, camera_);
            last_stats_.loop_inliers = static_cast<int>(est.inliers.size());
            // Verify geometrically: PnP from the old keyframe to this one. Only then trust it as a loop edge.
            if (est.success && last_stats_.loop_inliers >= params_.min_loop_inliers) {
                graph_.addEdge(c, id, est.motion, /*loop=*/true);
                ++loop_edges_;
                last_stats_.loop_accepted = true;
            }
        }
        loop_detector_->add(features.descriptors);
    }

    const int prev_id = keyframe_id_;   // the keyframe the associations come from
    keyframe_id_ = id;

    // make kf and push
    Keyframe kf;
    kf.features = std::move(features);
    kf.depth = depth.clone();
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(kf.features.keypoints, points);     // keypoints → plain (x, y) pixels
    kf.undistorted = undistortPixels(points, camera_);        // undistorted pixel per keypoint
    kf.map_point.assign(kf.features.keypoints.size(), -1);    // no map point yet for any keypoint
    keyframes_.push_back(std::move(kf));
    keyframe_pose_ = pose_;
    has_keyframe_ = true;

    Keyframe& new_kf = keyframes_[id];

    // Continue existing points: a keypoint PnP matched to a previous-keyframe keypoint
    // that already has a map point is another observation of that same point.
    for (const auto& [old_kp, new_kp] : associations) {
        const int m = keyframes_[prev_id].map_point[old_kp];
        if (m == -1) continue;                         // the old keypoint has no map point
        if (new_kf.map_point[new_kp] != -1) continue;  // already linked (two old keypoints
                                                    // matched the same new one)
        map_points_[m].observations.emplace_back(id, new_kp);
        new_kf.map_point[new_kp] = m;
    }

    // Create new points: every keypoint that's still unlinked and has a depth reading
    // becomes a new map point, placed in the world with this keyframe's pose.
    for (int k = 0; k < static_cast<int>(new_kf.map_point.size()); ++k) {
        if (new_kf.map_point[k] != -1) continue;       // linked in ①
        const std::optional<double> d = depthAt(new_kf.depth, new_kf.features.keypoints[k].pt);
        if (!d) continue;                              // no depth here: can't place it in 3D
        const Eigen::Vector3d X_cam = camera_.backproject(new_kf.undistorted[k], *d);
        new_kf.map_point[k] = static_cast<int>(map_points_.size());   // the new point's id
        map_points_.push_back(MapPoint{pose_ * X_cam, {{id, k}}});
    }

    // Everything stored per keyframe must have one entry per graph node, or an id would
    // point to the wrong keyframe's data.
    const int n = graph_.numNodes();
    if (static_cast<int>(keyframes_.size()) != n ||
        (loop_detector_ && loop_detector_->size() != n)) {
        throw std::logic_error("VisualOdometry: keyframe data and graph ids out of sync");
    }

    if (params_.local_ba) localBundleAdjust();
}

Pose VisualOdometry::track(const cv::Mat& gray, const cv::Mat& depth) {
    FrameFeatures current = detectOrb(gray);
    last_stats_ = TrackStats{};

    if (!has_keyframe_) {
        setKeyframe(std::move(current), depth);
        last_stats_.success = true;
    } else {
        // References into keyframes_: only valid until the next
        // setKeyframe() (push_back may reallocate). Don't use them after it.
        const Keyframe& kf = keyframes_[keyframe_id_];
        const FrameFeatures& kf_features = kf.features;
        const cv::Mat& kf_depth = kf.depth;

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
                std::vector<std::pair<int, int>> associations;
                for (int i : est.inliers) {
                    associations.emplace_back(matches[i].queryIdx, matches[i].trainIdx);
                }
                setKeyframe(std::move(current), depth, associations);
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

MapStats VisualOdometry::mapStats() const {
    MapStats result;
    result.points = static_cast<int>(map_points_.size());
    for (const MapPoint& mp : map_points_) {   // const&: read only, no copy of each point
        const int obs = static_cast<int>(mp.observations.size());
        result.observations += obs;
        if (obs >= 3) {
            ++result.points_3plus;
        }
    }

    return result;
}

void VisualOdometry::localBundleAdjust() {
    const auto start = std::chrono::steady_clock::now();   // for baSeconds()

    // 1. Which keyframes are free: the newest ba_window ones (never 0 or 1).
    const int n = static_cast<int>(keyframes_.size());
    const int first_free = std::max(2, n - params_.ba_window);
    if (first_free >= n) return;                                 // nothing to adjust yet

    // 2. The points seen by the free keyframes and observed at least twice in total
    //    (a point seen once gives BA nothing to compare).
    std::vector<int> point_ids;                       // map point ids in the problem (BA point order)
    std::unordered_map<int, int> point_to_ba;         // map point id -> BA point id
    for (int k = first_free; k < n; ++k) {
        for (int m : keyframes_[k].map_point) {
            if (m != -1 && map_points_[m].observations.size() >= 2 && !point_to_ba.count(m)) {
                point_to_ba[m] = static_cast<int>(point_ids.size());
                point_ids.push_back(m);
            }
        }
    }
    if (point_ids.empty()) return;

    // 3. Every keyframe that observes one of those points: the free ones, plus older
    //    (fixed) ones that add constraints. Sorted oldest first.
    std::vector<int> kf_ids;
    for (int m : point_ids) {
        for (const auto& [kf, kp] : map_points_[m].observations) {
            kf_ids.push_back(kf);
        }
    }
    std::sort(kf_ids.begin(), kf_ids.end());
    kf_ids.erase(std::unique(kf_ids.begin(), kf_ids.end()), kf_ids.end());
    std::unordered_map<int, int> kf_to_ba;            // keyframe id -> BA pose id
    for (int i = 0; i < static_cast<int>(kf_ids.size()); ++i) {
        kf_to_ba[kf_ids[i]] = i;
    }

    // 4. At least two fixed poses: pixels don't change if the whole scene is scaled, so one
    //    fixed pose can't pin down the scale. Points are continued only from the previous
    //    keyframe, so often only one keyframe outside the window sees them; then the oldest
    //    window keyframes in the problem are fixed too, until there are two.
    if (kf_ids.size() < 3) return;                    // need two fixed and at least one free
    int fix_below = first_free;                       // keyframes with id < fix_below don't move
    if (kf_ids[1] >= fix_below) {
        fix_below = kf_ids[1] + 1;
    }

    //    A window keyframe with only a few observations in the problem (e.g. the one right after
    //    a rejected frame: no associations to its predecessor) is barely connected, so the
    //    problem would be degenerate along its pose. It's fixed instead: its observations still
    //    constrain the points, it just doesn't move.
    std::vector<int> observation_count(kf_ids.size(), 0);
    for (int m : point_ids) {
        for (const auto& [kf, kp] : map_points_[m].observations) {
            ++observation_count[kf_to_ba.at(kf)];
        }
    }
    std::vector<bool> fixed(kf_ids.size());
    bool any_free = false;
    for (std::size_t i = 0; i < kf_ids.size(); ++i) {
        fixed[i] = kf_ids[i] < fix_below || observation_count[i] < params_.ba_min_observations;
        any_free = any_free || !fixed[i];
    }
    if (!any_free) return;

    // 5. Build the problem and solve.
    BundleAdjuster ba(camera_, params_.ba_depth_noise);
    for (std::size_t i = 0; i < kf_ids.size(); ++i) {
        ba.addPose(graph_.pose(kf_ids[i]), fixed[i]);
    }
    for (int m : point_ids) {
        ba.addPoint(map_points_[m].position);
    }
    for (int j = 0; j < static_cast<int>(point_ids.size()); ++j) {
        for (const auto& [kf, kp] : map_points_[point_ids[j]].observations) {
            // RGB-D: the measured depth at this keypoint (raw pixel), 0 if there's no reading.
            const Keyframe& k = keyframes_[kf];
            const double depth = depthAt(k.depth, k.features.keypoints[kp].pt).value_or(0.0);
            ba.addObservation(kf_to_ba.at(kf), j, k.undistorted[kp], depth);
        }
    }
    ba.optimize();

    // 6. Write back: free poses to the graph, points to the map, then the current state.
    for (int i = 0; i < static_cast<int>(kf_ids.size()); ++i) {
        if (!fixed[i]) graph_.setPose(kf_ids[i], ba.pose(i));
    }
    for (int j = 0; j < static_cast<int>(point_ids.size()); ++j) {
        map_points_[point_ids[j]].position = ba.point(j);
    }
    // The newest keyframe is in the window, so its pose may have changed; right now the
    // current frame is that keyframe. (trajectory() follows automatically: frames are stored
    // relative to their keyframe's graph pose.)
    keyframe_pose_ = graph_.pose(keyframe_id_);
    pose_ = keyframe_pose_;

    ++ba_runs_;
    ba_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace rtvm
