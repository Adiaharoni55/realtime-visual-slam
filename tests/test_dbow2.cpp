#include <vector>

#include <DBoW2/BowVector.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "rtvm/vocabulary.hpp"

namespace {

// Random ORB-like descriptors for one "image": each is a 1x32 CV_8U row, as DBoW2 expects.
std::vector<cv::Mat> randomDescriptors(cv::RNG& rng, int count) {
    std::vector<cv::Mat> descriptors;
    for (int i = 0; i < count; ++i) {
        cv::Mat d(1, 32, CV_8U);
        rng.fill(d, cv::RNG::UNIFORM, 0, 256);
        descriptors.push_back(d);
    }
    return descriptors;
}

}  // namespace

// Smoke test: DBoW2 links and works (train a tiny vocabulary, score an image with itself).
TEST(DBoW2Test, ImageMatchesItselfPerfectly) {
    cv::RNG rng(42);
    std::vector<std::vector<cv::Mat>> training;
    for (int image = 0; image < 5; ++image) {
        training.push_back(randomDescriptors(rng, 100));
    }

    // Plain TF weighting: with TF-IDF, a word seen in every training image gets idf = log(N/N) = 0
    // and is dropped, which on random data can empty the BoW vector depending on the platform's rand().
    rtvm::OrbVocabulary vocabulary(5, 2, DBoW2::TF, DBoW2::L1_NORM);   // branching 5, depth 2 -> up to 25 words
    vocabulary.create(training);
    ASSERT_GT(vocabulary.size(), 0u);

    DBoW2::BowVector v;
    vocabulary.transform(training[0], v);
    ASSERT_FALSE(v.empty());
    EXPECT_NEAR(vocabulary.score(v, v), 1.0, 1e-9);
}