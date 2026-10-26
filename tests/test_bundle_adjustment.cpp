#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "rtvm/bundle_adjustment.hpp"
#include "rtvm/camera.hpp"
#include "rtvm/pose.hpp"

// A camera with both rotation and translation, and one point in front of it.
// The point is chosen in the camera frame, so its true pixel comes from camera.project(),
// code we already trust; its world position comes from the pose (camera -> world).
class ReprojectionTest : public ::testing::Test {
protected:
    rtvm::PinholeCamera camera{525.0, 525.0, 319.5, 239.5};   // TUM default, no distortion

    // Translation (0.5, -0.2, 1.0) and a non-trivial rotation (unit quaternion, w last as in TUM).
    rtvm::Pose pose = rtvm::poseFromTum(0.5, -0.2, 1.0,
                                        0.1, 0.2, 0.05, std::sqrt(1.0 - 0.01 - 0.04 - 0.0025));

    Eigen::Vector3d point_cam{0.3, -0.1, 2.0};                       // 2 m in front of the camera
    Eigen::Vector3d point_world = pose * point_cam;                  // the same point in the world
    Eigen::Vector2d true_pixel = camera.project(point_cam);          // where it really appears
};

// Observed exactly where the point projects -> nothing to correct.
TEST_F(ReprojectionTest, ResidualIsZeroForPerfectProjection) {
    const Eigen::Vector2d residual =
        rtvm::reprojectionResidual(camera, pose, point_world, true_pixel);

    EXPECT_NEAR(residual.x(), 0.0, 1e-9);
    EXPECT_NEAR(residual.y(), 0.0, 1e-9);
}

// residual = projected - observed: observed (3, -2) px away from the projection -> residual (-3, 2).
TEST_F(ReprojectionTest, ResidualIsProjectedMinusObserved) {
    const Eigen::Vector2d observed = true_pixel + Eigen::Vector2d(3.0, -2.0);
    const Eigen::Vector2d residual =
        rtvm::reprojectionResidual(camera, pose, point_world, observed);

    EXPECT_NEAR(residual.x(), -3.0, 1e-9);
    EXPECT_NEAR(residual.y(), 2.0, 1e-9);
}

// A point 1 m behind the camera can't be projected.
TEST_F(ReprojectionTest, PointBehindCameraThrows) {
    const Eigen::Vector3d behind_world = pose * Eigen::Vector3d(0.0, 0.0, -1.0);
    EXPECT_THROW(rtvm::reprojectionResidual(camera, pose, behind_world, Eigen::Vector2d(0, 0)),
                 std::domain_error);
}
