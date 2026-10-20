#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rtvm/features.hpp"
#include "rtvm/tum_dataset.hpp"
#include "rtvm/camera.hpp"
#include "rtvm/motion.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: rtvm_match_demo <path to TUM sequence folder>\n";
        return 1;
    }
    std::string dir = argv[1];

    // Load and associate the sequence (only the image paths are needed here).
    auto rgb = rtvm::readTumFileList(dir + "/rgb.txt");
    auto depth = rtvm::readTumFileList(dir + "/depth.txt");
    auto gt = rtvm::readTumGroundTruth(dir + "/groundtruth.txt");
    auto frames = rtvm::associateTum(rgb, depth, gt);
    std::cout << "Associated frames: " << frames.size() << '\n';

    // Select two frames
    const auto& frame1 = frames[0];
    const auto& frame2 = frames[10];

    // How far did the camera truly move between the two frames? (from ground truth)
    rtvm::Pose motion = rtvm::relativePose(frame1.ground_truth.value(), frame2.ground_truth.value());
    std::cout << "True camera movement: " << motion.translation().norm() * 100 << " cm\n";
    std::cout << "True camera rotation: "
            << Eigen::AngleAxisd(motion.linear()).angle() * 180.0 / EIGEN_PI << " degrees\n";

    // Two frames a short time apart (10 frames = about 1/3 of a second).
    cv::Mat img_a = cv::imread(dir + "/" + frame1.rgb_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img_b = cv::imread(dir + "/" + frame2.rgb_path, cv::IMREAD_GRAYSCALE);

    rtvm::FrameFeatures fa = rtvm::detectOrb(img_a);
    rtvm::FrameFeatures fb = rtvm::detectOrb(img_b);
    std::vector<cv::DMatch> matches = rtvm::matchFeatures(fa, fb);

    // Load frame1's depth image and the camera
    cv::Mat depth_a = cv::imread(dir + "/" + frame1.depth_path, cv::IMREAD_UNCHANGED);
    rtvm::PinholeCamera cam = rtvm::loadCameraFromYaml("configs/tum_fr1.yaml");

    // Estimate the motion from frame1 to frame2.
    rtvm::MotionEstimate est = rtvm::estimateMotion(fa, fb, matches, depth_a, cam);
    std::cout << "Keypoints: " << fa.keypoints.size() << " and " << fb.keypoints.size()
              << ", good matches: " << matches.size() << '\n';
    std::cout << "PnP: success " << std::boolalpha << est.success
              << ", points with depth " << est.num_points
              << ", inliers " << est.inliers.size() << '\n';

    if (est.success) {
        std::cout << "Estimated movement: " << est.motion.translation().norm() * 100 << " cm, rotation: "
                  << Eigen::AngleAxisd(est.motion.linear()).angle() * 180.0 / EIGEN_PI << " degrees\n";

        // Error = what's left after "undoing" the true motion from the estimate (0 = perfect).
        rtvm::Pose error = motion.inverse() * est.motion;
        std::cout << "Error: " << error.translation().norm() * 100 << " cm, "
                  << Eigen::AngleAxisd(error.linear()).angle() * 180.0 / EIGEN_PI << " degrees\n";
    }

    // Draw both images side by side with lines between matched keypoints.
    cv::Mat out;
    std::filesystem::create_directories("output");
    cv::drawMatches(img_a, fa.keypoints, img_b, fb.keypoints, matches, out);
    cv::imwrite("output/matches.png", out);
    std::cout << "Saved output/matches.png\n";

    // Draws only the inliers
    std::vector<cv::DMatch> inlier_matches;
    for (int k : est.inliers){
        inlier_matches.push_back(matches[k]);
    }
    cv::drawMatches(img_a, fa.keypoints, img_b, fb.keypoints, inlier_matches, out);
    cv::imwrite("output/inliers.png", out);
    std::cout << "Saved output/inliers.png\n";

    return 0;
}