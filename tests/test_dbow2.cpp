#include <vector>

#include <DBoW2/BowVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace {

using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

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

    OrbVocabulary vocabulary(5, 2);   // branching factor 5, depth 2 -> up to 25 words
    vocabulary.create(training);
    ASSERT_GT(vocabulary.size(), 0u);

    DBoW2::BowVector v;
    vocabulary.transform(training[0], v);
    EXPECT_NEAR(vocabulary.score(v, v), 1.0, 1e-9);
}