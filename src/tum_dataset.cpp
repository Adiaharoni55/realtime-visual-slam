#include "rtvm/tum_dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iomanip>   

namespace {

// Index of the item whose timestamp is closest to t (items must not be empty).
// Works for any type with a .timestamp member (TimedFile, TimedPose).
template <typename T>
std::size_t closestIndex(const std::vector<T>& items, double t) {
    // Binary search: first item whose timestamp is >= t (items are sorted by time).
    auto it = std::lower_bound(items.begin(), items.end(), t,
                               [](const T& item, double value) { return item.timestamp < value; });

    if (it == items.begin()) return 0;                 // t is before every item
    if (it == items.end()) return items.size() - 1;    // t is after every item

    // The closest is either this item or the one just before it.
    std::size_t i = it - items.begin();
    return (t - items[i - 1].timestamp <= items[i].timestamp - t) ? i - 1 : i;
}

}  // namespace

namespace rtvm {

std::vector<TimedFile> readTumFileList(const std::string& path) {
    // Open the list file (rgb.txt or depth.txt).
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("readTumFileList: cannot open " + path);
    }

    // Each data line is "<timestamp> <image path>". Timestamps are needed later
    // to match each RGB image with its depth image and ground-truth pose.
    std::vector<TimedFile> files;
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comment lines.
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Split the line into its two fields; fail loudly on a malformed line.
        std::istringstream ss(line);
        double timestamp;
        std::string name;
        if (!(ss >> timestamp >> name)) {
            throw std::runtime_error("readTumFileList: bad line in " + path + ": " + line);
        }

        files.push_back({timestamp, name});
    }
    return files;
}

std::vector<TimedPose> readTumGroundTruth(const std::string& path){
    // Open the groundtruth file ("timestamp tx ty tz qx qy qz qw")
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("readTumGroundTruth: cannot open " + path);
    }

    // Each data line is "<timestamp> tx ty tz qx qy qz qw" (a pose from motion capture)
    std::vector<TimedPose> poses;
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comment lines.
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Split the line into its 8 fields; fail loudly on a malformed line.
        std::istringstream ss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            throw std::runtime_error("readTumGroundTruth: bad line in " + path + ": " + line);
        }

        poses.push_back({timestamp, poseFromTum(tx, ty, tz, qx, qy, qz, qw)});
    }
    return poses;   
}

void writeTumTrajectory(const std::string& path, const std::vector<TimedPose>& poses){
    // Open the file for writing 
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("writeTumTrajectory: cannot open " + path);
    }
    file << "# timestamp tx ty tz qx qy qz qw\n";
    file << std::fixed << std::setprecision(6);
    for (const TimedPose& p : poses) {
        Eigen::Quaterniond q(p.pose.linear());
        Eigen::Vector3d t = p.pose.translation();
        file << p.timestamp << ' '
            << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
            << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';    
    }
}


// A frame is kept when a depth image lies within max_diff seconds of its RGB image.
// Ground truth is attached only when a pose lies within max_diff; otherwise it stays empty.
std::vector<Frame> associateTum(const std::vector<TimedFile>& rgb,
                                const std::vector<TimedFile>& depth,
                                const std::vector<TimedPose>& ground_truth,
                                double max_diff) {
    std::vector<Frame> frames;

    // Nothing to match against: return no frames.
    if (depth.empty()) {
        return frames;
    }

    // closestIndex uses binary search, which needs time-sorted lists.
    // TUM files are sorted; fail loudly if one isn't.
    auto byTime = [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; };
    if (!std::is_sorted(depth.begin(), depth.end(), byTime) ||
        !std::is_sorted(ground_truth.begin(), ground_truth.end(), byTime)) {
        throw std::runtime_error("associateTum: depth and ground truth must be sorted by timestamp");
    }


    for (const TimedFile& image : rgb) {
        // Depth is required: skip this image if the closest depth is too far away in time.
        const TimedFile& d = depth[closestIndex(depth, image.timestamp)];
        if (std::abs(d.timestamp - image.timestamp) > max_diff) {
            continue;
        }

        // Start without ground truth.
        Frame frame{image.timestamp, image.path, d.path, std::nullopt};

        // Ground truth is optional: attach it only if a close pose exists.
        if (!ground_truth.empty()) {
            const TimedPose& g = ground_truth[closestIndex(ground_truth, image.timestamp)];
            if (std::abs(g.timestamp - image.timestamp) <= max_diff) {
                frame.ground_truth = g.pose;
            }
        }

        frames.push_back(frame);
    }
    return frames;
}

}  // namespace rtvm
