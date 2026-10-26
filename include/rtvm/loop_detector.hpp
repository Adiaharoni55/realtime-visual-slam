#pragma once

#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "rtvm/vocabulary.hpp"

namespace rtvm {

// An older keyframe that looks like the query: its id and BoW similarity score (0-1, higher = more similar).
struct LoopCandidate {
    int keyframe_id;
    double score;
};

// When a keyframe counts as a loop candidate.
struct LoopParams {
    int min_gap = 30;           // ignore the N most recent keyframes
    double min_score = 0.07;    // drop candidates scoring below this (chosen from the fr2/desk analysis)
};

// Finds loop-closure candidates: the older keyframe that looks most like a new one (bag of words).
// It only proposes candidates; geometric verification decides whether it's a real loop.
// For each new keyframe, call query() first, then add(), or the keyframe finds itself.
class LoopDetector {
public:
    // Loads the vocabulary file. Throws std::runtime_error if it doesn't exist.
    explicit LoopDetector(const std::string& vocabulary_path, LoopParams params = {});

    // Adds a keyframe (its Nx32 ORB descriptors).
    // Its id is the order of adding (0, 1, 2, ...), the same as the graph node id.
    void add(const cv::Mat& descriptors);

    // The most similar keyframe at least min_gap older than the newest one,
    // or empty if there are too few keyframes yet or the best score is below min_score.
    std::optional<LoopCandidate> query(const cv::Mat& descriptors) const;

    // How many keyframes were added.
    int size() const { return static_cast<int>(database_.size()); }

private:
    OrbDatabase database_;
    LoopParams params_;
};

} // namespace rtvm
