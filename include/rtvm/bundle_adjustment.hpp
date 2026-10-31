#pragma once

#include <vector>

#include <Eigen/Geometry>

#include "rtvm/camera.hpp"
#include "rtvm/pose.hpp"

namespace rtvm {

// Reprojection residual of one observation: where the world point lands when projected into
// the camera, minus the pixel where it was actually detected (residual = projected - observed,
// in pixels). camera_pose is camera -> world, as in the pose graph. No lens distortion: the
// observed pixel must already be undistorted.
// Exposed for testing; bundle adjustment uses the same computation.
// Throws std::domain_error if the point is behind (or too close to) the camera.
Eigen::Vector2d reprojectionResidual(const PinholeCamera& camera, const Pose& camera_pose,
                                     const Eigen::Vector3d& world_point,
                                     const Eigen::Vector2d& observed_pixel);

// Bundle adjustment: jointly adjusts camera poses and 3D points so that every point,
// projected into every camera that saw it, lands on its observed pixel.
// Fix at least two poses some distance apart: pixels don't change if the whole scene is scaled,
// so one fixed pose can't pin down the scale.
class BundleAdjuster {
public:
    // The intrinsics used for every observation.
    explicit BundleAdjuster(PinholeCamera camera);

    // Add a camera pose (camera -> world, initial guess). fixed = the optimizer may not move it.
    // Returns its id (0, 1, 2, ... in the order added).
    int addPose(const Pose& pose, bool fixed);

    // Add a 3D point in world coordinates (initial guess). Returns its id (0, 1, 2, ...).
    int addPoint(const Eigen::Vector3d& world);

    // "Pose pose_id saw point point_id at this pixel". The pixel must be undistorted.
    // depth: the depth camera's measurement of this point in meters, or 0 if there is none.
    // With depth, BA also pulls the point's depth in the camera toward the measurement
    // (RGB-D bundle adjustment); pixels alone constrain depth poorly when views are close.
    // Throws std::out_of_range for an id that doesn't exist.
    void addObservation(int pose_id, int point_id, const Eigen::Vector2d& pixel, double depth = 0.0);

    // Adjusts all non-fixed poses and all observed points to minimize the reprojection errors.
    // Huber loss: quadratic up to huber_delta pixels, linear beyond, so outliers count less.
    void optimize(double huber_delta = 2.0);

    // The current pose / point (optimized, once optimize() has run).
    // Throw std::out_of_range for an id that doesn't exist.
    Pose pose(int id) const;
    Eigen::Vector3d point(int id) const;

    // Sizes
    int numPoses() const { return static_cast<int>(poses_.size()); }
    int numPoints() const { return static_cast<int>(points_.size()); }
    int numObservations() const { return static_cast<int>(observations_.size()); }

private:
    // A pose as Ceres optimizes it: position and unit quaternion (camera -> world).
    struct PoseBlock {
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
        bool fixed;
    };

    // One pixel measurement: which camera saw which point, and where.
    struct Observation {
        int pose_id;
        int point_id;
        Eigen::Vector2d pixel;   // undistorted
        double depth;            // measured depth in meters, 0 = none
    };

    void checkPoseId(int id) const;    // throws std::out_of_range
    void checkPointId(int id) const;   // throws std::out_of_range

    PinholeCamera camera_;
    std::vector<PoseBlock> poses_;
    std::vector<Eigen::Vector3d> points_;
    std::vector<Observation> observations_;
};

}  // namespace rtvm
