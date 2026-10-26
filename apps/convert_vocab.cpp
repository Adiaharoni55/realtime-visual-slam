#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "rtvm/vocabulary.hpp"

// Converts a vocabulary between formats, e.g. the slow-loading YAML to our binary format:
//   rtvm_convert_vocab models/orb_vocab.yml.gz models/orb_vocab.bin
// The file extensions pick the formats (see saveVocabulary / loadVocabulary).
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input vocabulary> <output vocabulary>\n";
        return 1;
    }
    const std::string input = argv[1];
    const std::string output = argv[2];

    try {
        std::cout << "Loading " << input << "...\n";
        auto start = std::chrono::steady_clock::now();
        const rtvm::OrbVocabulary vocabulary = rtvm::loadVocabulary(input);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Loaded " << vocabulary.size() << " words in "
                  << std::chrono::duration<double>(end - start).count() << " s\n";

        // The output folder may not exist yet (e.g. models/ on a fresh clone).
        const std::filesystem::path out(output);
        if (out.has_parent_path()) {
            std::filesystem::create_directories(out.parent_path());
        }

        start = std::chrono::steady_clock::now();
        rtvm::saveVocabulary(vocabulary, output);
        end = std::chrono::steady_clock::now();
        std::cout << "Saved " << output << " in "
                  << std::chrono::duration<double>(end - start).count() << " s\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
