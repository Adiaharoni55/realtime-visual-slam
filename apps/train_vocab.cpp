#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "rtvm/features.hpp"
#include "rtvm/tum_dataset.hpp"
#include "rtvm/vocabulary.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output file> <TUM folder> [more TUM folders...]\n";
        return 1;
    }

    // Check every folder up front, so a typo in the last one fails now, not after minutes of work.
    for (int a = 2; a < argc; ++a) {
        std::string dir = argv[a];
        if (!std::filesystem::exists(dir + "/rgb.txt")) {
            std::cerr << "No rgb.txt in " << dir << '\n';
            return 1;
        }
    }

    std::string output_path = argv[1];
    std::vector<std::vector<cv::Mat>> training;
    std::size_t total_descriptors = 0;

    for (int a = 2; a < argc; ++a) {
        std::string dir = argv[a];
        auto files = rtvm::readTumFileList(dir + "/rgb.txt");
        // Every 5th image: varied views without near-duplicate frames.
        for (std::size_t i = 0; i < files.size(); i += 5) {
            std::string path = dir + "/" + files[i].path;
            cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (gray.empty()) {
                std::cerr << "Could not read " << path << '\n';
                continue;
            }
            rtvm::FrameFeatures features = rtvm::detectOrb(gray);
            std::vector<cv::Mat> rows = rtvm::descriptorRows(features.descriptors);
            if (!rows.empty()) {
                total_descriptors += rows.size();
                training.push_back(std::move(rows));
            }
        }
        std::cout << dir << ": " << training.size() << " images so far\n";
    }

    std::cout << training.size() << " images, " << total_descriptors << " descriptors\n";

    // The output folder may not exist yet (e.g. models/ on a fresh clone).
    const std::filesystem::path out(output_path);
    if (out.has_parent_path()) {
        std::filesystem::create_directories(out.parent_path());
    }

    // k-means tree: 10 branches per node, 5 levels -> up to 10^5 = 100,000 words.
    rtvm::OrbVocabulary vocabulary(10, 5);

    std::cout << "Training vocabulary (this takes a few minutes)...\n";
    const auto start = std::chrono::steady_clock::now();
    vocabulary.create(training);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Vocabulary: " << vocabulary.size() << " words, trained in "
              << seconds << " s\n";

    vocabulary.save(output_path);
    std::cout << "Saved " << output_path << '\n';

    return 0;
}
