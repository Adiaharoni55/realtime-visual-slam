#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

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

// Synthetic scene: 3 cameras in a row (x = 0, 0.2, 0.4 m, all looking along +z) and 50 random
// points 1.5-3 m in front, seen by all 3. Observations are the exact projections.
class BundleAdjusterTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (double x : {0.0, 0.2, 0.4}) {
            rtvm::Pose pose = rtvm::Pose::Identity();
            pose.translation() = Eigen::Vector3d(x, 0.0, 0.0);
            true_poses.push_back(pose);
        }
        cv::RNG rng(42);
        for (int j = 0; j < 50; ++j) {
            true_points.emplace_back(rng.uniform(-0.5, 0.9), rng.uniform(-0.4, 0.4),
                                     rng.uniform(1.5, 3.0));
        }
    }

    // Exact pixel of point j in camera i (world -> camera frame, then project).
    Eigen::Vector2d pixel(int i, int j) const {
        return camera.project(true_poses[i].inverse() * true_points[j]);
    }

    // Camera 2 off by ~5 cm and 2 degrees; every point off by up to 2 cm per coordinate.
    void perturb(std::vector<rtvm::Pose>& poses, std::vector<Eigen::Vector3d>& points) const {
        poses[2].translation() += Eigen::Vector3d(0.03, -0.03, 0.03);
        poses[2].linear() =
            poses[2].linear() * Eigen::AngleAxisd(2.0 * EIGEN_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
        cv::RNG rng(7);
        for (Eigen::Vector3d& p : points) {
            p += Eigen::Vector3d(rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02),
                                 rng.uniform(-0.02, 0.02));
        }
    }

    // Cameras 0 and 1 fixed (two fixed cameras pin down the scale), camera 2 and all points free.
    // outlier_fraction > 0: that share of the observations, picked at random (fixed seed), is
    // pushed 30 px to the right, like a bad match. Random, not "every n-th": with 50 points per
    // camera, every 10th would hit the same points in all 3 cameras, and a shift that is
    // consistent across cameras is explained away by moving the point (not an outlier at all).
    rtvm::BundleAdjuster makeAdjuster(const std::vector<rtvm::Pose>& start_poses,
                                      const std::vector<Eigen::Vector3d>& start_points,
                                      double outlier_fraction = 0.0) const {
        rtvm::BundleAdjuster ba(camera);
        for (std::size_t i = 0; i < start_poses.size(); ++i) {
            ba.addPose(start_poses[i], /*fixed=*/i < 2);
        }
        for (const Eigen::Vector3d& p : start_points) {
            ba.addPoint(p);
        }
        cv::RNG rng(123);
        for (int i = 0; i < static_cast<int>(start_poses.size()); ++i) {
            for (int j = 0; j < static_cast<int>(start_points.size()); ++j) {
                Eigen::Vector2d px = pixel(i, j);
                if (rng.uniform(0.0, 1.0) < outlier_fraction) px.x() += 30.0;
                ba.addObservation(i, j, px);
            }
        }
        return ba;
    }

    // How far camera 2 ended up from the truth (m).
    double camera2Error(const rtvm::BundleAdjuster& ba) const {
        return (ba.pose(2).translation() - true_poses[2].translation()).norm();
    }

    rtvm::PinholeCamera camera{525.0, 525.0, 319.5, 239.5};
    std::vector<rtvm::Pose> true_poses;
    std::vector<Eigen::Vector3d> true_points;
};

// The scene's pixels must all be inside the image, or the setup isn't realistic.
TEST_F(BundleAdjusterTest, SceneIsVisibleInAllCameras) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 50; ++j) {
            const Eigen::Vector2d px = pixel(i, j);
            EXPECT_TRUE(px.x() >= 0 && px.x() < 640 && px.y() >= 0 && px.y() < 480)
                << "camera " << i << ", point " << j << ": " << px.transpose();
        }
    }
}

// From a perturbed start, exact observations pull camera 2 and every point back to the truth.
TEST_F(BundleAdjusterTest, RecoversPerturbedPoseAndPoints) {
    std::vector<rtvm::Pose> start_poses = true_poses;
    std::vector<Eigen::Vector3d> start_points = true_points;
    perturb(start_poses, start_points);

    rtvm::BundleAdjuster ba = makeAdjuster(start_poses, start_points);
    ASSERT_EQ(ba.numObservations(), 150);
    ba.optimize();

    EXPECT_LT(camera2Error(ba), 1e-3);
    const double angle_error = Eigen::AngleAxisd(ba.pose(2).rotation().transpose() *
                                                 true_poses[2].rotation()).angle();
    EXPECT_LT(angle_error, 1e-3);   // radians (~0.06 degrees)
    EXPECT_LT((ba.pose(0).translation() - true_poses[0].translation()).norm(), 1e-12);   // fixed
    for (int j = 0; j < 50; ++j) {
        EXPECT_LT((ba.point(j) - true_points[j]).norm(), 1e-3) << "point " << j;
    }
}

// 10% of the observations are 30 px off. Plain squared error (a huge delta: never switches to
// linear) is pulled further away than where it started (~6 cm); Huber stays much closer (~1.6 cm).
// Huber reduces outliers' influence but doesn't remove it: with only 3 views per point, a bad
// observation can't be outvoted strongly. Removing outliers entirely is a separate step.
TEST_F(BundleAdjusterTest, HuberLossResistsOutliers) {
    std::vector<rtvm::Pose> start_poses = true_poses;
    std::vector<Eigen::Vector3d> start_points = true_points;
    perturb(start_poses, start_points);

    rtvm::BundleAdjuster huber = makeAdjuster(start_poses, start_points, /*outlier_fraction=*/0.1);
    huber.optimize();   // delta = 2 px
    rtvm::BundleAdjuster plain = makeAdjuster(start_poses, start_points, /*outlier_fraction=*/0.1);
    plain.optimize(1e6);

    const double error_huber = camera2Error(huber);
    const double error_plain = camera2Error(plain);
    EXPECT_LT(error_huber, 2e-2) << "plain: " << error_plain;
    EXPECT_GT(error_plain, 3.0 * error_huber)
        << "huber: " << error_huber << " m, plain: " << error_plain << " m";
}

TEST_F(BundleAdjusterTest, InvalidIdsThrow) {
    rtvm::BundleAdjuster ba(camera);
    ba.addPose(rtvm::Pose::Identity(), true);
    ba.addPoint(Eigen::Vector3d(0, 0, 2));
    const Eigen::Vector2d px(320, 240);

    EXPECT_NO_THROW(ba.addObservation(0, 0, px));
    EXPECT_THROW(ba.addObservation(5, 0, px), std::out_of_range);    // no pose 5
    EXPECT_THROW(ba.addObservation(-1, 0, px), std::out_of_range);
    EXPECT_THROW(ba.addObservation(0, 5, px), std::out_of_range);    // no point 5
    EXPECT_THROW(ba.addObservation(0, -1, px), std::out_of_range);
    EXPECT_THROW(ba.pose(1), std::out_of_range);
    EXPECT_THROW(ba.point(1), std::out_of_range);
}

// With one fixed pose, pixels alone can't tell the scale: scaling every other camera and every
// point about camera 0 leaves all pixels unchanged. Measured depths pin it down.
TEST_F(BundleAdjusterTest, DepthPinsScaleWithOneFixedPose) {
    // Start: cameras 1 and 2 and all points at 1.1x the true scale (camera 0 is at the origin),
    // so every pixel is still exact and only the depths disagree.
    std::vector<rtvm::Pose> start_poses = true_poses;
    std::vector<Eigen::Vector3d> start_points = true_points;
    for (int i = 1; i < 3; ++i) start_poses[i].translation() *= 1.1;
    for (Eigen::Vector3d& p : start_points) p *= 1.1;

    auto solve = [&](bool with_depth) {
        rtvm::BundleAdjuster ba(camera);
        for (int i = 0; i < 3; ++i) ba.addPose(start_poses[i], /*fixed=*/i == 0);
        for (const Eigen::Vector3d& p : start_points) ba.addPoint(p);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 50; ++j) {
                const double depth = (true_poses[i].inverse() * true_points[j]).z();
                ba.addObservation(i, j, pixel(i, j), with_depth ? depth : 0.0);
            }
        }
        ba.optimize();
        return ba;
    };

    const rtvm::BundleAdjuster with_depth = solve(true);
    EXPECT_LT(camera2Error(with_depth), 1e-3);
    for (int j = 0; j < 50; ++j) {
        EXPECT_LT((with_depth.point(j) - true_points[j]).norm(), 1e-3) << "point " << j;
    }

    // Without depth the wrong scale fits the pixels perfectly, so nothing moves:
    // camera 2 stays ~4 cm off (0.4 m x 0.1).
    const rtvm::BundleAdjuster pixels_only = solve(false);
    EXPECT_GT(camera2Error(pixels_only), 1e-2);
}
