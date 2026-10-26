#include "rtvm/loop_detector.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

#include "rtvm/features.hpp"
#include "rtvm/vocabulary.hpp"

namespace {

// Load a vocabulary file. k, L, weighting and scoring all come from the file.
rtvm::OrbVocabulary loadVocabulary(const std::string& path) {
    // DBoW2's load() throws a plain std::string on failure, so check first for a clear error.
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("LoopDetector: vocabulary not found: " + path);
    }
    rtvm::OrbVocabulary vocabulary;
    vocabulary.load(path);
    return vocabulary;
}

}  // namespace

namespace rtvm {

// The database keeps its own copy of the vocabulary; the loaded one is a temporary, freed here.
// No direct index (false, 0): we don't need it for feature matching.
LoopDetector::LoopDetector(const std::string& vocabulary_path, LoopParams params)
    : database_(loadVocabulary(vocabulary_path), false, 0), params_(params) {}

void LoopDetector::add(const cv::Mat& descriptors) {
    // The database turns the descriptors into a bag of words (a sparse histogram of visual words)
    // with its own vocabulary. Ids are 0, 1, 2, ... in the order added, like the graph node ids.
    database_.add(descriptorRows(descriptors));
}

std::optional<LoopCandidate> LoopDetector::query(const cv::Mat& descriptors) const {
    // Only keyframes at least min_gap older than the newest: recent ones always look alike.
    const int max_id = size() - 1 - params_.min_gap;
    // Below we pass max_id + 1, and DBoW2 reads -1 as "no limit": it would search the recent ones too.
    if (max_id < 0) return std::nullopt;

    // The single best-scoring keyframe among ids 0..max_id (one match for the whole image).
    // + 1: DBoW2 keeps ids < max_id (its doc says <=, the code says <).
    DBoW2::QueryResults results;
    database_.query(descriptorRows(descriptors), results, 1, max_id + 1);
    if (results.empty()) return std::nullopt;

    return LoopCandidate{static_cast<int>(results[0].Id), results[0].Score};
}

}  // namespace rtvm
