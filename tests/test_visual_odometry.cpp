#include <gtest/gtest.h>

#include <cmath>

#include "rtvm/pose.hpp"
#include "rtvm/visual_odometry.hpp"

namespace {

// A pose rotated by angle_deg around axis, then shifted by t.
rtvm::Pose makePose(const Eigen::Vector3d& t, const Eigen::Vector3d& axis, double angle_deg) {
    rtvm::Pose pose = rtvm::Pose::Identity();
    pose.linear() = Eigen::AngleAxisd(angle_deg * M_PI / 180.0, axis.normalized()).toRotationMatrix();
    pose.translation() = t;
    return pose;
}

// Odometry's prediction in all tests: not the identity, so the check must compare the two poses,
// not just look at the measured one.
const rtvm::Pose kPredicted = makePose({0.5, -0.2, 1.0}, {0, 1, 0}, 30.0);

// The measured loop = the prediction, off by a known disagreement.
rtvm::Pose measuredWith(const rtvm::Pose& disagreement) { return kPredicted * disagreement; }

}  // namespace

// PnP and odometry agree exactly -> accepted, even with no path in between.
TEST(LoopConsistencyTest, IdenticalPoseIsAccepted) {
    EXPECT_TRUE(rtvm::loopConsistentWithOdometry(kPredicted, kPredicted, 0.0, rtvm::VoParams{}));
}

// The false loop 74 -> 106 on structure_texture_far: 2.37 m and 46.6° off after ~2 m of path
// (limits there: 0.4 m and 16°) -> rejected.
TEST(LoopConsistencyTest, FalseLoopFromRepeatingTextureIsRejected) {
    const rtvm::Pose measured = measuredWith(makePose({2.37, 0.0, 0.0}, {0.3, 1, 0.2}, 46.6));
    EXPECT_FALSE(rtvm::loopConsistentWithOdometry(kPredicted, measured, 2.0, rtvm::VoParams{}));
}

// fr2/desk: after an 18 m loop, odometry has drifted 30 cm and ~7° (limits: 2.0 m and 64°) -> accepted.
TEST(LoopConsistencyTest, DriftAfterLongLoopIsAccepted) {
    const rtvm::Pose measured = measuredWith(makePose({0.2, 0.1, 0.2}, {0, 0, 1}, 7.0));   // 0.30 m
    EXPECT_TRUE(rtvm::loopConsistentWithOdometry(kPredicted, measured, 18.0, rtvm::VoParams{}));
}

// The tolerance grows with the path: the same 30 cm is too much right after the old keyframe.
TEST(LoopConsistencyTest, SameDriftAfterShortPathIsRejected) {
    const rtvm::Pose measured = measuredWith(makePose({0.2, 0.1, 0.2}, {0, 0, 1}, 7.0));
    EXPECT_FALSE(rtvm::loopConsistentWithOdometry(kPredicted, measured, 0.5, rtvm::VoParams{}));
}

// Translation and rotation are checked separately: either one alone can reject a loop.
TEST(LoopConsistencyTest, RotationAloneCanReject) {
    const rtvm::Pose measured = measuredWith(makePose({0.05, 0.0, 0.0}, {1, 0, 0}, 25.0));
    EXPECT_FALSE(rtvm::loopConsistentWithOdometry(kPredicted, measured, 2.0, rtvm::VoParams{}));
}

TEST(LoopConsistencyTest, TranslationAloneCanReject) {
    const rtvm::Pose measured = measuredWith(makePose({0.6, 0.0, 0.0}, {1, 0, 0}, 1.0));
    EXPECT_FALSE(rtvm::loopConsistentWithOdometry(kPredicted, measured, 2.0, rtvm::VoParams{}));
}
