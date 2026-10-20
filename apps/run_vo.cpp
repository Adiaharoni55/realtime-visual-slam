#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "rtvm/camera.hpp"
#include "rtvm/tum_dataset.hpp"
#include "rtvm/visual_odometry.hpp"

int main(int argc, char** argv) {
    // 2 required arguments + up to 3 optional ones.
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <TUM folder> <camera yaml>"
                  << " [max_translation_m] [max_rotation_deg] [min_inliers]\n";
        return 1;
    }

    std::string dir = argv[1];
    rtvm::PinholeCamera camera = rtvm::loadCameraFromYaml(argv[2]);

    // Optional thresholds: use the given value if it's there, otherwise the default.
        rtvm::VoParams params;   // starts with the defaults from the struct
    if (argc > 3) params.max_translation  = std::stod(argv[3]);
    if (argc > 4) params.max_rotation_deg = std::stod(argv[4]);
    if (argc > 5) params.min_inliers      = std::stoi(argv[5]);

    std::cout << "Keyframe thresholds: " << params.max_translation << " m, "
              << params.max_rotation_deg << " deg, " << params.min_inliers << " inliers\n";


    auto frames = rtvm::associateTum(rtvm::readTumFileList(dir + "/rgb.txt"),
                                     rtvm::readTumFileList(dir + "/depth.txt"),
                                     rtvm::readTumGroundTruth(dir + "/groundtruth.txt"));
    std::cout << "Frames: " << frames.size() << '\n';

    rtvm::VisualOdometry vo(camera, params);
    std::vector<rtvm::TimedPose> trajectory;

    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < frames.size(); ++i) {
        cv::Mat gray = cv::imread(dir + "/" + frames[i].rgb_path, cv::IMREAD_GRAYSCALE);
        cv::Mat depth = cv::imread(dir + "/" + frames[i].depth_path, cv::IMREAD_UNCHANGED);

        rtvm::Pose pose = vo.track(gray, depth);
        trajectory.push_back({frames[i].timestamp, pose});

        if (i % 100 == 0) {
            std::cout << "Frame " << i << " / " << frames.size() << '\n';
        }
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::filesystem::create_directories("output");
    rtvm::writeTumTrajectory("output/trajectory.txt", trajectory);

    std::cout << "Failed frames: " << vo.failures() << '\n'
              << "Keyframes: " << vo.keyframes() << "\n"
              << "Average time per frame: " << ms / frames.size() << " ms\n"
              << "Saved output/trajectory.txt\n";
    
    return 0;
}