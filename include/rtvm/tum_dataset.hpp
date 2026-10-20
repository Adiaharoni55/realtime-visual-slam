#pragma once

#include "rtvm/pose.hpp"
#include <string>
#include <vector>
#include <optional>

namespace rtvm {

// One line of rgb.txt or depth.txt: a timestamp and an image path.
struct TimedFile {
    double timestamp;   // seconds
    std::string path;   // relative to the sequence folder, e.g. "rgb/1305031102.175304.png"
};

// Read rgb.txt or depth.txt. Skips comment lines (#). Throws if the file can't be opened.
std::vector<TimedFile> readTumFileList(const std::string& path);

// One line of groundtruth.txt: a timestamp and the true camera pose.
struct TimedPose {
    double timestamp;   // seconds
    Pose pose;          // camera pose in the world (motion-capture) frame
};

// Read groundtruth.txt ("timestamp tx ty tz qx qy qz qw"). Skips comment lines (#).
// Throws if the file can't be opened or a line is malformed.
std::vector<TimedPose> readTumGroundTruth(const std::string& path);

// Write poses in TUM format ("timestamp tx ty tz qx qy qz qw"), one line per pose.
// Throws if the file can't be created.
void writeTumTrajectory(const std::string& path, const std::vector<TimedPose>& poses);

// One synchronized frame: an RGB image, its depth image and the true pose.
struct Frame {
    double timestamp;         // RGB image time (seconds)
    std::string rgb_path;
    std::string depth_path;
    std::optional<Pose> ground_truth;
};

// Match each RGB image with the closest depth image and ground-truth pose.
// Frames with no match within max_diff seconds are skipped.
std::vector<Frame> associateTum(const std::vector<TimedFile>& rgb,
                                const std::vector<TimedFile>& depth,
                                const std::vector<TimedPose>& ground_truth,
                                double max_diff = 0.02);

}  // namespace rtvm