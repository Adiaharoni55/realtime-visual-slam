#pragma once

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/TemplatedDatabase.h>

namespace rtvm {

// DBoW2 vocabulary of ORB descriptors (each descriptor is a 1×32 CV_8U cv::Mat).
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

// database type of ORB descriptors
using OrbDatabase = DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

}  // namespace rtvm