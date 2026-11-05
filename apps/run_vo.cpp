#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rtvm/camera.hpp"
#include "rtvm/tum_dataset.hpp"
#include "rtvm/visual_odometry.hpp"
#include "rtvm/segmentation.hpp"

namespace {

void printUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " <TUM folder> <camera yaml>"
              << " [max_translation_m] [max_rotation_deg] [min_inliers]"
              << " [--vocab <file>] [--ba]"
              << " [--mask <model.onnx>] [--start <frame>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Split the arguments: "--vocab <file>", "--mask <file>", "--start <frame>" and "--ba" are
    // named (anywhere), all others are positional, in order.
    std::vector<std::string> positional;
    std::string vocab_path;   // empty = no loop detection
    std::string mask_path;    // empty = no masking
    bool local_ba = false;    // --ba: local bundle adjustment after each keyframe
    int start_frame = 0;      // --start: skip the frames before this one (for repeated runs)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vocab") {
            if (i + 1 >= argc) {   // no file after --vocab
                printUsage(argv[0]);
                return 1;
            }
            vocab_path = argv[++i];   // ++i: the path is consumed here, not as a positional
        } else if (arg == "--mask") {
            if (i + 1 >= argc) {   // no file after --mask
                printUsage(argv[0]);
                return 1;
            }
            mask_path = argv[++i];
        } else if (arg == "--start") {
            if (i + 1 >= argc) {   // no frame after --start
                printUsage(argv[0]);
                return 1;
            }
            start_frame = std::stoi(argv[++i]);
        } else if (arg == "--ba") {
            local_ba = true;          // a flag: no value follows
        } else {
            positional.push_back(arg);
        }
    }

    // 2 required arguments + up to 3 optional ones.
    if (positional.size() < 2 || positional.size() > 5) {
        printUsage(argv[0]);
        return 1;
    }
    if (!vocab_path.empty() && !std::filesystem::exists(vocab_path)) {
        std::cerr << "Vocabulary not found: " << vocab_path << '\n';
        return 1;
    }

    if (!mask_path.empty() && !std::filesystem::exists(mask_path)) {
        std::cerr << "Mask not found: " << mask_path << '\n';
        return 1;
    }

    const std::filesystem::path output_dir = "output";
    std::filesystem::create_directories(output_dir);

    std::string dir = positional[0];
    rtvm::PinholeCamera camera = rtvm::loadCameraFromYaml(positional[1]);

    // Optional thresholds: use the given value if it's there, otherwise the default.
    rtvm::VoParams params;   // starts with the defaults from the struct
    if (positional.size() > 2) params.max_translation  = std::stod(positional[2]);
    if (positional.size() > 3) params.max_rotation_deg = std::stod(positional[3]);
    if (positional.size() > 4) params.min_inliers      = std::stoi(positional[4]);
    params.local_ba = local_ba;

    std::cout << "Keyframe thresholds: " << params.max_translation << " m, "
              << params.max_rotation_deg << " deg, " << params.min_inliers << " inliers\n";
    std::cout << "Local BA: "
              << (params.local_ba ? "on (window " + std::to_string(params.ba_window) + ")" : "off")
              << '\n';

    auto frames = rtvm::associateTum(rtvm::readTumFileList(dir + "/rgb.txt"),
                                     rtvm::readTumFileList(dir + "/depth.txt"),
                                     rtvm::readTumGroundTruth(dir + "/groundtruth.txt"));
    // A later start gives different keyframes on almost the same data: repeated runs with
    // start 0..4 show how much the results vary by chance.
    if (start_frame < 0 || start_frame >= static_cast<int>(frames.size())) {
        std::cerr << "Start frame " << start_frame << " is outside the sequence (" << frames.size()
                  << " frames)\n";
        return 1;
    }
    frames.erase(frames.begin(), frames.begin() + start_frame);
    std::cout << "Start frame: " << start_frame << '\n';
    std::cout << "Frames: " << frames.size() << '\n';

    rtvm::VisualOdometry vo(camera, params);
    if (!vocab_path.empty()) {
        std::cout << "Loading vocabulary " << vocab_path << "...\n";
        vo.enableLoopDetection(vocab_path);
    }

    std::vector<rtvm::TimedPose> trajectory;
    std::ofstream csv(output_dir / "stats.csv");
    if (!csv) {
        std::cerr << "Could not open " << (output_dir / "stats.csv") << '\n';
        return 1;
    }
    // The person segmenter (optional) and the kernel that enlarges its mask: the model's edges
    // are imprecise, and keypoints right at a person's border are the most likely to be wrong.
    std::unique_ptr<rtvm::PersonSegmenter> segmenter;
    if (!mask_path.empty()) {
        std::cout << "Masking people with " << mask_path << '\n';
        segmenter = std::make_unique<rtvm::PersonSegmenter>(mask_path);
    }
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

    csv << std::fixed << std::setprecision(6);
    csv << "timestamp,matches,inliers,success,new_keyframe,loop_candidate,loop_score,loop_inliers,loop_accepted,rejected,masked,loop_inconsistent\n";

    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < frames.size(); ++i) {
        cv::Mat gray = cv::imread(dir + "/" + frames[i].rgb_path, cv::IMREAD_GRAYSCALE);
        cv::Mat depth = cv::imread(dir + "/" + frames[i].depth_path, cv::IMREAD_UNCHANGED);

        cv::Mat static_mask;   // empty = features everywhere
        double masked = 0.0;   // fraction of the image that is masked
        if (segmenter) {
            const cv::Mat color = cv::imread(dir + "/" + frames[i].rgb_path, cv::IMREAD_COLOR);
            cv::Mat person = segmenter->segment(color);       // 255 = person
            cv::dilate(person, person, kernel);               // cover the edges too
            static_mask = 255 - person;                       // 255 = static background
            masked = 1.0 - static_cast<double>(cv::countNonZero(static_mask)) / static_mask.total();
        }
        rtvm::Pose pose = vo.track(gray, depth, static_mask);

        const auto& s = vo.lastStats();
        double timestamp = frames[i].timestamp;
        csv << timestamp        << ','
            << s.matches        << ','
            << s.inliers        << ','
            << s.success        << ','
            << s.new_keyframe   << ','
            << s.loop_candidate << ','
            << s.loop_score     << ','
            << s.loop_inliers   << ','
            << s.loop_accepted  << ','
            << s.rejected       << ','
            << masked           << ','
            << s.loop_inconsistent << '\n';

        trajectory.push_back({timestamp, pose});

        if (i % 100 == 0) {
            std::cout << "Frame " << i << " / " << frames.size() << '\n';
        }
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    rtvm::writeTumTrajectory((output_dir / "trajectory_vo.txt").string(), trajectory);

    // Check: the poses rebuilt from the graph must match the ones track() returned
    // (nothing was optimized yet, so any difference means the bookkeeping is wrong).
    // With local BA, keyframes move after they were tracked, so the difference is BA's correction.
    const std::vector<rtvm::Pose> from_graph = vo.trajectory();
    if (from_graph.size() != trajectory.size()) {
        std::cerr << "Graph trajectory has " << from_graph.size() << " poses, expected "
                  << trajectory.size() << '\n';
        return 1;
    }
    double max_diff = 0.0;
    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        const double diff =
            (from_graph[i].translation() - trajectory[i].pose.translation()).norm();
        max_diff = std::max(max_diff, diff);
    }

    // Optimize the pose graph: the loop edges pull the drifted poses back.
    if (vo.loopEdges() > 0) {
        std::cout << "Optimizing pose graph...\n";
        vo.optimize();
    }

    // The corrected trajectory: poses from the (optimized) graph, with the original timestamps.
    const std::vector<rtvm::Pose> optimized = vo.trajectory();
    std::vector<rtvm::TimedPose> optimized_trajectory;
    optimized_trajectory.reserve(trajectory.size());
    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        optimized_trajectory.push_back({trajectory[i].timestamp, optimized[i]});
    }
    rtvm::writeTumTrajectory((output_dir / "trajectory.txt").string(), optimized_trajectory);

    // Every accepted loop edge, to compare with the ground truth (scripts/check_loops.py).
    std::ofstream loops_csv(output_dir / "loops.csv");
    loops_csv << std::fixed << std::setprecision(6);
    loops_csv << "from,to,inliers,tx,ty,tz,qx,qy,qz,qw\n";
    for (const rtvm::LoopRecord& loop : vo.loops()) {
        const Eigen::Vector3d t = loop.measured.translation();
        const Eigen::Quaterniond q(loop.measured.rotation());
        loops_csv << loop.from << ',' << loop.to << ',' << loop.inliers << ','
                  << t.x() << ',' << t.y() << ',' << t.z() << ','
                  << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
    }

    const rtvm::MapStats map = vo.mapStats();

    std::cout << "Graph: " << vo.graph().numNodes() << " nodes, " << vo.graph().numEdges() << " edges ("
              << vo.loopEdges() << " loops)\n"
              << (params.local_ba ? "Max BA correction (graph vs. tracked): "
                                  : "Max difference graph vs. tracked: ")
              << max_diff << " m\n"
              << "Failed frames: " << vo.failures() << '\n'
              << "Average time per frame: " << ms / frames.size() << " ms\n";
    if (params.local_ba) {
        // BA runs inside track() (synchronously), so it's part of the time per frame above.
        std::cout << "Local BA: " << vo.baRuns() << " runs, " << vo.baSeconds() << " s total ("
                  << (vo.baRuns() > 0 ? 1000.0 * vo.baSeconds() / vo.baRuns() : 0.0)
                  << " ms per run, " << 1000.0 * vo.baSeconds() / frames.size()
                  << " ms per frame on average)\n";
    }
    std::cout
              << "Saved " << (output_dir / "trajectory_vo.txt").string() << ", "
              << (output_dir / "trajectory.txt").string() << " and "
              << (output_dir / "stats.csv").string() << '\n'
              << "Map: " << map.points << " points, " << map.observations << " observations ("
              << (map.points > 0 ? double(map.observations) / map.points : 0.0)
              << " per point), " << map.points_3plus << " seen by 3+ keyframes\n";

    return 0;
}
