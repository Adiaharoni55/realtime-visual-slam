#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "rtvm/features.hpp"
#include "rtvm/loop_detector.hpp"
#include "rtvm/vocabulary.hpp"

namespace {

// Random ORB-like "images": each is an Nx32 CV_8U descriptor matrix, as detectOrb returns.
std::vector<cv::Mat> randomImages(int count, int descriptors_per_image = 100) {
    cv::RNG rng(42);
    std::vector<cv::Mat> images;
    for (int i = 0; i < count; ++i) {
        cv::Mat m(descriptors_per_image, 32, CV_8U);
        rng.fill(m, cv::RNG::UNIFORM, 0, 256);
        images.push_back(m);
    }
    return images;
}

}  // namespace

// CI has no real vocabulary file, so each test trains a tiny one on its images and saves it.
class LoopDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        images_ = randomImages(40);

        std::vector<std::vector<cv::Mat>> training;
        for (const cv::Mat& image : images_) {
            training.push_back(rtvm::descriptorRows(image));
        }
        // Plain TF weighting, as in the DBoW2 smoke test (TF-IDF can zero out words on random data).
        rtvm::OrbVocabulary vocabulary(5, 2, DBoW2::TF, DBoW2::L1_NORM);
        vocabulary.create(training);

        // One file per test: ctest may run tests in parallel, and they'd share a single path.
        const std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        vocab_path_ = (std::filesystem::temp_directory_path() / ("rtvm_test_vocab_" + name + ".yml.gz")).string();
        vocabulary.save(vocab_path_);
    }

    void TearDown() override { std::filesystem::remove(vocab_path_); }

    // Add the first `count` images as keyframes 0..count-1.
    void addImages(rtvm::LoopDetector& detector, int count) const {
        for (int i = 0; i < count; ++i) {
            detector.add(images_[i]);
        }
    }

    std::vector<cv::Mat> images_;
    std::string vocab_path_;
};

// An image already in the database (and old enough) is its own best match.
TEST_F(LoopDetectorTest, FindsSameImageAmongOldKeyframes) {
    rtvm::LoopDetector detector(vocab_path_);
    addImages(detector, 40);
    ASSERT_EQ(detector.size(), 40);

    std::optional<rtvm::LoopCandidate> candidate = detector.query(images_[3]);
    ASSERT_TRUE(candidate);
    EXPECT_EQ(candidate->keyframe_id, 3);
    EXPECT_NEAR(candidate->score, 1.0, 1e-9);
}

// The newest keyframe must not match itself or anything within the last min_gap (30) keyframes.
TEST_F(LoopDetectorTest, IgnoresRecentKeyframes) {
    rtvm::LoopDetector detector(vocab_path_);
    addImages(detector, 40);

    std::optional<rtvm::LoopCandidate> candidate = detector.query(images_[39]);
    ASSERT_TRUE(candidate);
    EXPECT_LE(candidate->keyframe_id, 39 - 30);
}

// The oldest allowed keyframe (newest id - min_gap = 9) is still searched.
TEST_F(LoopDetectorTest, SearchesUpToTheGap) {
    rtvm::LoopDetector detector(vocab_path_);
    addImages(detector, 40);

    std::optional<rtvm::LoopCandidate> candidate = detector.query(images_[9]);
    ASSERT_TRUE(candidate);
    EXPECT_EQ(candidate->keyframe_id, 9);
}

// Too few keyframes: nothing is old enough. 29 is the trap: the limit passed to DBoW2
// lands exactly on its -1 = "search everything".
TEST_F(LoopDetectorTest, NoCandidateWhileTooFewKeyframes) {
    for (int count : {10, 29, 30}) {
        rtvm::LoopDetector detector(vocab_path_);
        addImages(detector, count);
        EXPECT_FALSE(detector.query(images_[count - 1])) << count << " keyframes";
    }

    // With 31, keyframe 0 is exactly min_gap older than the newest, so it's allowed.
    rtvm::LoopDetector detector(vocab_path_);
    addImages(detector, 31);
    std::optional<rtvm::LoopCandidate> candidate = detector.query(images_[0]);
    ASSERT_TRUE(candidate);
    EXPECT_EQ(candidate->keyframe_id, 0);
}

TEST_F(LoopDetectorTest, MissingVocabularyThrows) {
    EXPECT_THROW(rtvm::LoopDetector("does/not/exist.yml.gz"), std::runtime_error);
}
