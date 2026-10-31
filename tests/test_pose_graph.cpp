#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "rtvm/pose.hpp"
#include "rtvm/pose_graph.hpp"

namespace {

// n poses on a circle (radius in meters), turning 360/n degrees around z at each step,
// so the last pose is one step away from the first: a closed loop with a known answer.
std::vector<rtvm::Pose> circlePoses(int n = 8, double radius = 1.0) {
    std::vector<rtvm::Pose> poses;
    for (int k = 0; k < n; ++k) {
        const double angle = 2.0 * EIGEN_PI * k / n;
        rtvm::Pose pose = rtvm::Pose::Identity();
        pose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        pose.translation() = radius * Eigen::Vector3d(std::cos(angle), std::sin(angle), 0.0);
        poses.push_back(pose);
    }
    return poses;
}

}  // namespace

TEST(PoseGraphTest, AddNodeRoundTrip) {
    rtvm::PoseGraph graph;
    const rtvm::Pose a = rtvm::poseFromTum(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9);
    const rtvm::Pose b = rtvm::poseFromTum(-1.0, 0.5, 0.0, 0.0, 0.0, 0.7071, 0.7071);

    EXPECT_EQ(graph.addNode(a), 0);
    EXPECT_EQ(graph.addNode(b), 1);
    EXPECT_EQ(graph.numNodes(), 2);
    EXPECT_TRUE(graph.pose(0).isApprox(a, 1e-9));
    EXPECT_TRUE(graph.pose(1).isApprox(b, 1e-9));
}

TEST(PoseGraphTest, InvalidIdThrows) {
    rtvm::PoseGraph graph;
    graph.addNode(rtvm::Pose::Identity());   // only id 0 exists

    EXPECT_THROW(graph.pose(-1), std::out_of_range);
    EXPECT_THROW(graph.pose(1), std::out_of_range);   // == numNodes(): one past the end
    EXPECT_THROW(graph.addEdge(0, 1, rtvm::Pose::Identity()), std::out_of_range);
}

// Edges that agree exactly with the poses leave nothing to fix.
TEST(PoseGraphTest, ChainOnlyKeepsPoses) {
    const std::vector<rtvm::Pose> truth = circlePoses();
    const int n = static_cast<int>(truth.size());

    rtvm::PoseGraph graph;
    for (const rtvm::Pose& pose : truth) {
        graph.addNode(pose);
    }
    for (int k = 0; k + 1 < n; ++k) {
        graph.addEdge(k, k + 1, rtvm::relativePose(truth[k], truth[k + 1]));
    }

    graph.optimize();

    for (int k = 0; k < n; ++k) {
        EXPECT_TRUE(graph.pose(k).isApprox(truth[k], 1e-6)) << "node " << k;
    }
}

// Odometry with a small error on every step drifts; one correct loop edge
// lets the optimizer spread that error over the whole loop.
TEST(PoseGraphTest, LoopEdgeReducesDrift) {
    const std::vector<rtvm::Pose> truth = circlePoses();
    const int n = static_cast<int>(truth.size());

    // The same small error on every step (2 degrees, 1 cm), like systematic drift.
    rtvm::Pose step_error = rtvm::Pose::Identity();
    step_error.linear() =
        Eigen::AngleAxisd(2.0 * EIGEN_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    step_error.translation() = Eigen::Vector3d(0.01, 0.0, 0.0);

    // Build the graph the way the tracker would: chain the noisy measurements.
    rtvm::PoseGraph graph;
    rtvm::Pose current = truth[0];
    graph.addNode(current);
    for (int k = 0; k + 1 < n; ++k) {
        const rtvm::Pose noisy = rtvm::relativePose(truth[k], truth[k + 1]) * step_error;
        current = current * noisy;
        graph.addNode(current);
        graph.addEdge(k, k + 1, noisy);
    }

    // Loop closure: the last node, measured correctly relative to the first.
    graph.addEdge(n - 1, 0, rtvm::relativePose(truth[n - 1], truth[0]));

    auto maxPositionError = [&]() {
        double worst = 0.0;
        for (int k = 0; k < n; ++k) {
            const double error = (graph.pose(k).translation() - truth[k].translation()).norm();
            worst = std::max(worst, error);
        }
        return worst;
    };

    const double before = maxPositionError();
    graph.optimize();
    const double after = maxPositionError();

    std::cout << "Largest position error: before " << before * 100 << " cm, after "
              << after * 100 << " cm\n";
    EXPECT_LT(after, 0.5 * before);
}

// setPose replaces one node's pose (e.g. with a bundle adjustment result) and nothing else.
TEST(PoseGraphTest, SetPoseOverwritesNode) {
    rtvm::PoseGraph graph;
    graph.addNode(rtvm::Pose::Identity());
    graph.addNode(rtvm::Pose::Identity());

    const double w = std::sqrt(1.0 - 0.01 - 0.04 - 0.0025);
    const rtvm::Pose p = rtvm::poseFromTum(0.3, -0.1, 0.5, 0.1, 0.2, 0.05, w);
    graph.setPose(1, p);

    EXPECT_TRUE(graph.pose(1).isApprox(p, 1e-12));
    EXPECT_TRUE(graph.pose(0).isApprox(rtvm::Pose::Identity(), 1e-12));   // untouched
    EXPECT_THROW(graph.setPose(5, p), std::out_of_range);
    EXPECT_THROW(graph.setPose(-1, p), std::out_of_range);
}

// After a pose moves, chain edges are re-measured from the poses; a loop edge keeps its measurement.
TEST(PoseGraphTest, RefreshChainEdgesFollowsPoses) {
    auto at_x = [](double x) {
        rtvm::Pose pose = rtvm::Pose::Identity();
        pose.translation() = Eigen::Vector3d(x, 0.0, 0.0);
        return pose;
    };
    rtvm::PoseGraph graph;
    for (double x : {0.0, 1.0, 2.0}) {
        graph.addNode(at_x(x));
    }
    graph.addEdge(0, 1, rtvm::relativePose(at_x(0.0), at_x(1.0)));   // chain
    graph.addEdge(1, 2, rtvm::relativePose(at_x(1.0), at_x(2.0)));   // chain
    const rtvm::Pose loop_measured = at_x(-2.1);                      // as if PnP measured it
    graph.addEdge(2, 0, loop_measured, /*loop=*/true);

    graph.setPose(1, at_x(1.5));   // e.g. moved by bundle adjustment
    graph.refreshChainEdges();

    EXPECT_FALSE(graph.edge(0).loop);
    EXPECT_FALSE(graph.edge(1).loop);
    EXPECT_TRUE(graph.edge(2).loop);
    EXPECT_TRUE(graph.edge(0).measured.isApprox(
        rtvm::relativePose(graph.pose(0), graph.pose(1)), 1e-12));   // now 1.5 m, not 1.0
    EXPECT_TRUE(graph.edge(1).measured.isApprox(
        rtvm::relativePose(graph.pose(1), graph.pose(2)), 1e-12));   // now 0.5 m
    EXPECT_TRUE(graph.edge(2).measured.isApprox(loop_measured, 1e-12));   // loop: unchanged
    EXPECT_THROW(graph.edge(3), std::out_of_range);
}
