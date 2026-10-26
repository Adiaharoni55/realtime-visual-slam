#include "rtvm/vocabulary.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace rtvm {

namespace {

constexpr char kMagic[8] = {'R', 'T', 'V', 'M', 'V', 'O', 'C', '1'};

template <typename T>
void writeValue(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
T readValue(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

bool isBinaryPath(const std::string& path) {
    return std::filesystem::path(path).extension() == ".bin";
}

// The tree lives in DBoW2's protected members, so only a subclass can read and write it.
class BinaryVocabulary : public OrbVocabulary {
public:
    BinaryVocabulary() = default;
    explicit BinaryVocabulary(const OrbVocabulary& vocabulary) : OrbVocabulary(vocabulary) {}

    // Layout: magic, k, L, weighting, scoring, node count, word count, then per node:
    // id, parent, weight, word id, children (count + ids, in order), descriptor (length + bytes).
    void writeTo(std::ostream& out) const {
        out.write(kMagic, sizeof(kMagic));
        writeValue<std::int32_t>(out, m_k);
        writeValue<std::int32_t>(out, m_L);
        writeValue<std::int32_t>(out, static_cast<std::int32_t>(m_weighting));
        writeValue<std::int32_t>(out, static_cast<std::int32_t>(m_scoring));
        writeValue<std::uint32_t>(out, static_cast<std::uint32_t>(m_nodes.size()));
        writeValue<std::uint32_t>(out, static_cast<std::uint32_t>(m_words.size()));

        for (const Node& node : m_nodes) {
            writeValue<std::uint32_t>(out, node.id);
            writeValue<std::uint32_t>(out, node.parent);
            writeValue<double>(out, node.weight);
            writeValue<std::uint32_t>(out, node.word_id);
            // Children in their original order: transform() breaks distance ties by order.
            writeValue<std::uint32_t>(out, static_cast<std::uint32_t>(node.children.size()));
            for (DBoW2::NodeId child : node.children) {
                writeValue<std::uint32_t>(out, child);
            }
            // The root has no descriptor; every other node has 32 bytes.
            const std::uint8_t length = node.descriptor.empty() ? 0 : 32;
            writeValue<std::uint8_t>(out, length);
            out.write(reinterpret_cast<const char*>(node.descriptor.data), length);
        }
    }

    void readFrom(std::istream& in, const std::string& path) {
        char magic[sizeof(kMagic)];
        in.read(magic, sizeof(magic));
        if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
            throw std::runtime_error("loadVocabulary: not an rtvm vocabulary file: " + path);
        }
        m_k = readValue<std::int32_t>(in);
        m_L = readValue<std::int32_t>(in);
        m_weighting = static_cast<DBoW2::WeightingType>(readValue<std::int32_t>(in));
        m_scoring = static_cast<DBoW2::ScoringType>(readValue<std::int32_t>(in));
        const auto node_count = readValue<std::uint32_t>(in);
        const auto word_count = readValue<std::uint32_t>(in);

        m_nodes.assign(node_count, Node());
        for (Node& node : m_nodes) {
            node.id = readValue<std::uint32_t>(in);
            node.parent = readValue<std::uint32_t>(in);
            node.weight = readValue<double>(in);
            node.word_id = readValue<std::uint32_t>(in);
            const auto child_count = readValue<std::uint32_t>(in);
            node.children.resize(child_count);
            for (DBoW2::NodeId& child : node.children) {
                child = readValue<std::uint32_t>(in);
            }
            const auto length = readValue<std::uint8_t>(in);
            if (length == 32) {
                node.descriptor.create(1, 32, CV_8U);
                in.read(reinterpret_cast<char*>(node.descriptor.data), length);
            } else if (length != 0) {
                throw std::runtime_error("loadVocabulary: bad descriptor length in " + path);
            }
            if (!in) break;   // truncated file, reported below
        }
        if (!in) {
            throw std::runtime_error("loadVocabulary: file is truncated: " + path);
        }

        // m_words points into m_nodes, so it's rebuilt now that m_nodes has its final size.
        m_words.assign(word_count, nullptr);
        for (std::size_t i = 1; i < m_nodes.size(); ++i) {   // skip the root
            Node& node = m_nodes[i];
            if (node.isLeaf()) {
                if (node.word_id >= word_count) {
                    throw std::runtime_error("loadVocabulary: bad word id in " + path);
                }
                m_words[node.word_id] = &node;
            }
        }

        createScoringObject();   // for the scoring type just read
    }
};

}  // namespace

void saveVocabulary(const OrbVocabulary& vocabulary, const std::string& path) {
    if (!isBinaryPath(path)) {
        vocabulary.save(path);
        return;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("saveVocabulary: cannot create " + path);
    }
    BinaryVocabulary(vocabulary).writeTo(out);
    if (!out) {
        throw std::runtime_error("saveVocabulary: write failed: " + path);
    }
}

OrbVocabulary loadVocabulary(const std::string& path) {
    // DBoW2's load() throws a plain std::string on failure, so check first for a clear error.
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("loadVocabulary: vocabulary not found: " + path);
    }
    if (!isBinaryPath(path)) {
        OrbVocabulary vocabulary;
        vocabulary.load(path);
        return vocabulary;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loadVocabulary: cannot open " + path);
    }
    BinaryVocabulary vocabulary;
    vocabulary.readFrom(in, path);
    return vocabulary;   // copied into an OrbVocabulary
}

}  // namespace rtvm
