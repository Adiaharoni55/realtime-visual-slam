#include <gtest/gtest.h>

#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rtvm/features.hpp"

// Two crops of the same texture, offset by (10, 5) px: matches must show that offset.
TEST(FeaturesTest, MatchesRecoverKnownShift) {
    // A random but repeatable texture (fixed seed), slightly blurred so corners are stable.
    cv::setRNGSeed(42);
    cv::Mat texture(260, 340, CV_8UC1);
    cv::randu(texture, 0, 255);
    cv::GaussianBlur(texture, texture, cv::Size(5, 5), 1.5);

    // Frame a starts at (0, 0); frame b starts at (10, 5) in the same texture.
    // So a point at (x, y) in a appears at (x - 10, y - 5) in b.
    cv::Mat a = texture(cv::Rect(0, 0, 320, 240)).clone();
    cv::Mat b = texture(cv::Rect(10, 5, 320, 240)).clone();

    rtvm::FrameFeatures fa = rtvm::detectOrb(a);
    rtvm::FrameFeatures fb = rtvm::detectOrb(b);
    std::vector<cv::DMatch> matches = rtvm::matchFeatures(fa, fb);

    ASSERT_GT(matches.size(), 50u);   // plenty of matches on a rich texture

    // Count matches whose displacement is the expected (-10, -5), within 1 px.
    int correct = 0;
    for (const cv::DMatch& m : matches) {
        cv::Point2f pa = fa.keypoints[m.queryIdx].pt;
        cv::Point2f pb = fb.keypoints[m.trainIdx].pt;
        if (std::abs((pb.x - pa.x) - (-10.0f)) < 1.0f && std::abs((pb.y - pa.y) - (-5.0f)) < 1.0f) {
            ++correct;
        }
    }
    // at least 85% of matches are right
    EXPECT_GT(correct, 0.85 * matches.size())
        << "correct: " << correct << " of " << matches.size() << " matches";
}

// A 3x32 descriptor matrix becomes 3 Mats of 1x32, each with the bytes of its original row.
TEST(FeaturesTest, DescriptorRowsSplitsMatrixIntoRows) {
    cv::Mat m(3, 32, CV_8U);
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            m.at<uchar>(r, c) = static_cast<uchar>(r * 32 + c);   // different in every row
        }
    }

    std::vector<cv::Mat> rows = rtvm::descriptorRows(m);

    ASSERT_EQ(rows.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(rows[i].rows, 1);
        EXPECT_EQ(rows[i].cols, 32);
        EXPECT_EQ(rows[i].type(), CV_8U);
        EXPECT_EQ(cv::norm(rows[i], m.row(i), cv::NORM_INF), 0.0);   // same bytes
        EXPECT_EQ(rows[i].data, m.ptr(i));   // a view into m, not a copy
    }
}

// No descriptors (e.g. an image with no keypoints) -> no rows.
TEST(FeaturesTest, DescriptorRowsOfEmptyMatrixIsEmpty) {
    EXPECT_TRUE(rtvm::descriptorRows(cv::Mat()).empty());
}
