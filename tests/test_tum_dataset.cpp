#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <filesystem>
#include "rtvm/tum_dataset.hpp"

// Reading the sample list keeps the 2 data lines and skips the 3 comment lines.
TEST(TumDatasetTest, ReadsFileListSkippingComments) {
    std::string path = std::string(RTVM_TEST_DATA_DIR) + "/rgb_sample.txt";
    std::vector<rtvm::TimedFile> files = rtvm::readTumFileList(path);

    // Stop here if the count is wrong: the checks below read files[0].
    ASSERT_EQ(files.size(), 2u);

    EXPECT_NEAR(files[0].timestamp, 1305031102.175304, 1e-6);
    EXPECT_EQ(files[0].path, "rgb/1305031102.175304.png");
}

// A list file that doesn't exist throws instead of returning an empty list.
TEST(TumDatasetTest, MissingFileListThrows) {
    std::string path = std::string(RTVM_TEST_DATA_DIR) + "/no_such_file.txt";
    EXPECT_THROW(rtvm::readTumFileList(path), std::runtime_error);
}

// Reading the sample ground truth keeps the 2 data lines and parses each into a pose
TEST(TumDatasetTest, ReadsGroundTruthSkippingComments) {
    std::string path = std::string(RTVM_TEST_DATA_DIR) + "/groundtruth_sample.txt";
    std::vector<rtvm::TimedPose> poses = rtvm::readTumGroundTruth(path);

    // Stop here if the count is wrong: the checks below read poses[0].
    ASSERT_EQ(poses.size(), 2u);
    
    EXPECT_NEAR(poses[0].timestamp, 1305031098.6659, 1e-6);
    EXPECT_NEAR(poses[0].pose.translation().x(), 1.3563, 1e-9);
    EXPECT_NEAR(poses[0].pose.translation().y(), 0.6305, 1e-9);
    EXPECT_NEAR(poses[0].pose.translation().z(), 1.6380, 1e-9);
    
}

// A ground truth file that doesn't exist throws instead of returning an empty list.
TEST(TumDatasetTest, MissingGroundTruthThrows) {
    std::string path = std::string(RTVM_TEST_DATA_DIR) + "/no_such_file.txt";
    EXPECT_THROW(rtvm::readTumGroundTruth(path), std::runtime_error);
}

// Each RGB image gets the depth image and pose with the closest timestamps.
TEST(TumDatasetTest, AssociatesClosestDepthAndPose) {
    // RGB at t = 1.000
    std::vector<rtvm::TimedFile> rgb = {{1.000, "rgb/a.png"}};
    // Depth at 0.990 (10 ms away) and 1.005 (5 ms away): 1.005 is closer
    std::vector<rtvm::TimedFile> depth = {{0.990, "depth/x.png"}, {1.005, "depth/y.png"}};
    // Poses at 0.995 (5 ms away, at x = 1) and 1.020 (20 ms away, at x = 2): 0.995 is closer
    std::vector<rtvm::TimedPose> gt = {
        {0.995, rtvm::poseFromTum(1, 0, 0, 0, 0, 0, 1)},
        {1.020, rtvm::poseFromTum(2, 0, 0, 0, 0, 0, 1)},
    };

    std::vector<rtvm::Frame> frames = rtvm::associateTum(rgb, depth, gt);

    ASSERT_EQ(frames.size(), 1u);
    EXPECT_EQ(frames[0].rgb_path, "rgb/a.png");
    EXPECT_EQ(frames[0].depth_path, "depth/y.png");
    ASSERT_TRUE(frames[0].ground_truth.has_value());
    EXPECT_NEAR(frames[0].ground_truth->translation().x(), 1.0, 1e-9);
}

// An RGB image with no depth image within max_diff is skipped.
TEST(TumDatasetTest, SkipsFrameWithoutCloseDepth){
    std::vector<rtvm::TimedFile> rgb = {{1.000, "rgb/a.png"}, {2.000, "rgb/b.png"}};
    std::vector<rtvm::TimedFile> depth = {{1.01, "depth/x.png"}};
        std::vector<rtvm::TimedPose> gt = {
        {1.000, rtvm::poseFromTum(1, 0, 0, 0, 0, 0, 1)},
        {2.000, rtvm::poseFromTum(2, 0, 0, 0, 0, 0, 1)} 
    };

    std::vector<rtvm::Frame> frames = rtvm::associateTum(rgb, depth, gt);

    ASSERT_EQ(frames.size(), 1u);
    EXPECT_EQ(frames[0].rgb_path, "rgb/a.png");
    ASSERT_TRUE(frames[0].ground_truth.has_value());
    EXPECT_NEAR(frames[0].ground_truth->translation().x(), 1.0, 1e-9);
}

// Unsorted input is rejected, because binary search would give wrong matches.
TEST(TumDatasetTest, RejectsUnsortedInput){
    std::vector<rtvm::TimedFile> rgb = {{1.000, "rgb/a.png"}};
    std::vector<rtvm::TimedFile> depth = {{1.005, "depth/y.png"}, {0.990, "depth/x.png"}};
    std::vector<rtvm::TimedPose> gt = {{1.000, rtvm::poseFromTum(1, 0, 0, 0, 0, 0, 1)}};
    EXPECT_THROW(rtvm::associateTum(rgb, depth, gt), std::runtime_error);
}

// Writing a trajectory and reading it back gives the same timestamps and positions.
TEST(TumDatasetTest, WriteThenReadGivesSamePoses) {
    // Two poses: positions (1,2,3) and (4,5,6), no rotation
    std::vector<rtvm::TimedPose> poses = {
        {1.5, rtvm::poseFromTum(1, 2, 3, 0, 0, 0, 1)},
        {2.5, rtvm::poseFromTum(4, 5, 6, 0, 0, 0, 1)},
    };
    std::string path = (std::filesystem::temp_directory_path() / "rtvm_traj.txt").string();

    // Round trip: write, then read back
    rtvm::writeTumTrajectory(path, poses);
    std::vector<rtvm::TimedPose> read = rtvm::readTumGroundTruth(path);

    ASSERT_EQ(read.size(), 2u);
    EXPECT_NEAR(read[1].timestamp, 2.5, 1e-6);
    EXPECT_NEAR(read[1].pose.translation().z(), 6.0, 1e-6);
}

// Ground truth only scores the result; it must never decide which frames are tracked.
// (On fr2/desk, motion-capture gaps were silently dropping ~700 frames.)
TEST(TumDatasetTest, KeepsFrameWithoutCloseGroundTruth) {
    std::vector<rtvm::TimedFile> rgb = {{1.000, "rgb/a.png"}};
    std::vector<rtvm::TimedFile> depth = {{1.002, "depth/x.png"}};
        std::vector<rtvm::TimedPose> gt = {
        {2.000, rtvm::poseFromTum(1, 0, 0, 0, 0, 0, 1)}
    };

    std::vector<rtvm::Frame> frames = rtvm::associateTum(rgb, depth, gt);

    ASSERT_EQ(frames.size(), 1u);
    EXPECT_EQ(frames[0].rgb_path, "rgb/a.png");
    EXPECT_FALSE(frames[0].ground_truth.has_value());
}

// Sequences without ground truth (e.g. your own camera) must still produce frames.
TEST(TumDatasetTest, WorksWithEmptyGroundTruth) {
    std::vector<rtvm::TimedFile> rgb = {{1.000, "rgb/a.png"}, {2.000, "rgb/b.png"}};
    std::vector<rtvm::TimedFile> depth = {{1.01, "depth/x.png"}, {2.01, "depth/y.png"}};
    std::vector<rtvm::TimedPose> gt = {};

    std::vector<rtvm::Frame> frames = rtvm::associateTum(rgb, depth, gt);

    ASSERT_EQ(frames.size(), 2u);
    EXPECT_EQ(frames[0].rgb_path, "rgb/a.png");
    EXPECT_FALSE(frames[0].ground_truth.has_value());
    EXPECT_EQ(frames[1].rgb_path, "rgb/b.png");
    EXPECT_FALSE(frames[1].ground_truth.has_value());
}
