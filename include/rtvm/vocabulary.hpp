#pragma once

#include <string>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/TemplatedDatabase.h>

namespace rtvm {

// DBoW2 vocabulary of ORB descriptors (each descriptor is a 1×32 CV_8U cv::Mat).
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

// database type of ORB descriptors
using OrbDatabase = DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

// Save / load a vocabulary. The file name picks the format: ".bin" is our own binary format
// (loads in about a second), anything else is DBoW2's YAML (e.g. ".yml.gz", minutes for 100k words).
// Both throw std::runtime_error if the file can't be written / doesn't exist or is invalid.
void saveVocabulary(const OrbVocabulary& vocabulary, const std::string& path);
OrbVocabulary loadVocabulary(const std::string& path);

}  // namespace rtvm
