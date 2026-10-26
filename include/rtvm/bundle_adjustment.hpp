#pragma once

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

}  // namespace rtvm
