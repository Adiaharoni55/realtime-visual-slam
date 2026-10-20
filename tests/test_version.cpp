#include <regex>
#include <string>

#include <gtest/gtest.h>

#include "rtvm/version.hpp"

// The version comes from CMake's project(VERSION ...), so the test checks its format
// instead of repeating the number.
TEST(VersionTest, HasMajorMinorPatchFormat) {
    const std::string v = rtvm::version();
    EXPECT_TRUE(std::regex_match(v, std::regex(R"(\d+\.\d+\.\d+)")))
        << "Unexpected version string: '" << v << "'";
}