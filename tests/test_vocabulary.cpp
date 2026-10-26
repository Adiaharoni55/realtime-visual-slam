#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <DBoW2/BowVector.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "rtvm/features.hpp"
#include "rtvm/vocabulary.hpp"

namespace {

// Random ORB-like "images": each is a list of 1x32 CV_8U descriptors, as DBoW2 expects.
std::vector<std::vector<cv::Mat>> randomImages(int count, int descriptors_per_image = 100) {
    cv::RNG rng(7);
    std::vector<std::vector<cv::Mat>> images;
    for (int i = 0; i < count; ++i) {
        cv::Mat m(descriptors_per_image, 32, CV_8U);
        rng.fill(m, cv::RNG::UNIFORM, 0, 256);
        images.push_back(rtvm::descriptorRows(m));
    }
    return images;
}

// The bag of words of every image must be identical with both vocabularies (word ids and weights).
void expectSameBagsOfWords(const rtvm::OrbVocabulary& a, const rtvm::OrbVocabulary& b,
                           const std::vector<std::vector<cv::Mat>>& images) {
    for (std::size_t i = 0; i < images.size(); ++i) {
        DBoW2::BowVector va, vb;
        a.transform(images[i], va);
        b.transform(images[i], vb);
        ASSERT_FALSE(va.empty());
        EXPECT_EQ(va, vb) << "image " << i;
    }
}

}  // namespace

// Trains a small vocabulary (k=5, L=3) and gives each test its own temp file prefix.
class VocabularyTest : public ::testing::Test {
protected:
    void SetUp() override {
        training_ = randomImages(30);
        // TF-IDF (the default) as in the real vocabulary; with 3 levels no word is in every image.
        vocabulary_ = rtvm::OrbVocabulary(5, 3);
        vocabulary_.create(training_);
        const std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        prefix_ = (std::filesystem::temp_directory_path() / ("rtvm_test_vocab_" + name)).string();
    }

    void TearDown() override {
        for (const char* ext : {".bin", ".yml.gz"}) {
            std::filesystem::remove(prefix_ + ext);
        }
    }

    std::vector<std::vector<cv::Mat>> training_;
    rtvm::OrbVocabulary vocabulary_;
    std::string prefix_;
};

// Save + load in binary gives the same vocabulary: same settings, same words for every image.
TEST_F(VocabularyTest, BinaryRoundTripGivesSameBagOfWords) {
    rtvm::saveVocabulary(vocabulary_, prefix_ + ".bin");
    const rtvm::OrbVocabulary loaded = rtvm::loadVocabulary(prefix_ + ".bin");

    ASSERT_GT(vocabulary_.size(), 0u);
    EXPECT_EQ(loaded.size(), vocabulary_.size());
    EXPECT_EQ(loaded.getBranchingFactor(), vocabulary_.getBranchingFactor());
    EXPECT_EQ(loaded.getDepthLevels(), vocabulary_.getDepthLevels());
    EXPECT_EQ(loaded.getWeightingType(), vocabulary_.getWeightingType());
    EXPECT_EQ(loaded.getScoringType(), vocabulary_.getScoringType());

    expectSameBagsOfWords(vocabulary_, loaded, training_);
    expectSameBagsOfWords(vocabulary_, loaded, randomImages(10, 200));   // unseen images too
}

// The real conversion path: YAML-loaded vocabulary -> .bin -> same bags of words as the YAML one.
TEST_F(VocabularyTest, BinaryMatchesYamlLoad) {
    rtvm::saveVocabulary(vocabulary_, prefix_ + ".yml.gz");
    const rtvm::OrbVocabulary from_yaml = rtvm::loadVocabulary(prefix_ + ".yml.gz");
    rtvm::saveVocabulary(from_yaml, prefix_ + ".bin");
    const rtvm::OrbVocabulary from_binary = rtvm::loadVocabulary(prefix_ + ".bin");

    expectSameBagsOfWords(from_yaml, from_binary, training_);
}

TEST_F(VocabularyTest, NotAVocabularyFileThrows) {
    std::ofstream(prefix_ + ".bin") << "hello, not a vocabulary";
    EXPECT_THROW(rtvm::loadVocabulary(prefix_ + ".bin"), std::runtime_error);
}

// A cut-off file (e.g. an interrupted copy) must fail loudly, not load half a tree.
TEST_F(VocabularyTest, TruncatedFileThrows) {
    rtvm::saveVocabulary(vocabulary_, prefix_ + ".bin");
    const auto size = std::filesystem::file_size(prefix_ + ".bin");
    std::filesystem::resize_file(prefix_ + ".bin", size / 2);
    EXPECT_THROW(rtvm::loadVocabulary(prefix_ + ".bin"), std::runtime_error);
}

TEST_F(VocabularyTest, MissingFileThrows) {
    EXPECT_THROW(rtvm::loadVocabulary(prefix_ + ".bin"), std::runtime_error);
}
