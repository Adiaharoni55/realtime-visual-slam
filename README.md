# Real-Time Visual SLAM

[![CI](https://github.com/Adiaharoni55/realtime-visual-slam/actions/workflows/ci.yml/badge.svg)](https://github.com/Adiaharoni55/realtime-visual-slam/actions/workflows/ci.yml)

A real-time C++17 SLAM system that estimates motion and builds a 3D map from a camera and a LiDAR. A deep-learning segmentation model (ONNX Runtime) masks moving objects such as people and cars, so they don't corrupt tracking.

> **Status: v0.1, visual odometry.** RGB-D visual odometry runs faster than real time (~15 ms/frame) and is evaluated on three TUM sequences. Next: bundle adjustment and loop closure.

## Progress

| Step | Status |
| --- | --- |
| 1. Project setup: CMake, sanitizers, unit tests, CI on Linux / macOS / Windows | ✅ Done |
| 2. Camera geometry: pinhole model, poses and quaternions, YAML camera config | ✅ Done |
| 3. Visual odometry on TUM RGB-D | ✅ Done (v0.1) |
| 4. Backend: bundle adjustment, pose graph, loop closure | Planned |
| 5. Masking moving objects with a deep-learning model (ONNX Runtime) | Planned |
| 6. LiDAR + camera fusion on KITTI | Planned |
| 7. Real-time performance: multithreading, profiling | Planned |
| 8. Docker, ROS 2 node, visualization | Planned |
| 9. Final results and evaluation | Planned |

## What works today

- **Pinhole camera model:** project 3D points to pixels and back-project pixels with depth to 3D, with input validation.
- **Pose utilities:** TUM-format poses (translation + quaternion) to 4×4 rigid transforms, and relative motion between poses.
- **Camera config:** intrinsics loaded from YAML files in `configs/`.
- **TUM RGB-D loader:** reads image lists and ground truth, and associates RGB and depth by timestamp (binary search, validated input). Ground truth is optional, so it never decides which frames get tracked.
- **ORB features:** detection and matching with Lowe's ratio test.
- **Motion estimation:** PnP + RANSAC on depth-backprojected points, with keypoint undistortion.
- **Visual odometry:** frame-to-keyframe tracking (new keyframe after 5 cm, 5° or too few inliers), writing a TUM-format trajectory and per-frame statistics.
- **Evaluation:** ATE/RPE with evo on three sequences, a threshold sweep, and a [failure analysis](docs/failure-analysis.md).
- **Unit tests** for all of the above (GoogleTest, including synthetic scenes with known motion), run in CI on every push.

## Results

RGB-D visual odometry, no backend optimization yet. Release build, Apple Silicon. Evaluated with [evo](https://github.com/MichaelGrupp/evo) (`--align`; RPE over 1-second windows).

| TUM sequence | Path | Frames | ATE | RPE | Speed | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| fr1/xyz | short, back and forth | 796 | 5.00 cm | 2.34 cm/s | 15.1 ms/frame | 0 |
| fr2/desk | 18.9 m loop around a desk | 2,965 | 13.3 cm | 1.62 cm/s | 14.8 ms/frame | 0 |
| fr1/desk | fast handheld | 596 | 6.49 cm | 3.97 cm/s | 15.1 ms/frame | 0 |

Keyframes vs. frame-to-frame tracking on fr2/desk: **ATE 45.8 → 13.3 cm**, RPE 4.33 → 1.62 cm/s.

![Estimated trajectory vs. ground truth on fr2/desk (top view, colored by error)](docs/images/fr2_desk_map.png)

The remaining error is mostly slow drift: the loop doesn't fully close. That's what the backend (Step 4) addresses. Where and why tracking struggles is described in the [failure analysis](docs/failure-analysis.md).

### Reproduce

```bash
cmake --preset release && cmake --build --preset release
./build/release/rtvm_run_vo data/rgbd_dataset_freiburg2_desk configs/tum_fr2.yaml   # optional: [max_m] [max_deg] [min_inliers]
evo_ape tum data/rgbd_dataset_freiburg2_desk/groundtruth.txt output/trajectory.txt --align
evo_rpe tum data/rgbd_dataset_freiburg2_desk/groundtruth.txt output/trajectory.txt --align --delta 30 --delta_unit f --all_pairs
python scripts/plot_stats.py   # inliers and keyframes over time
```

## Planned pipeline

1. Read camera frames (TUM RGB-D, KITTI) and LiDAR scans (KITTI).
2. Segment moving objects with an ONNX model and build a mask.
3. Detect and match ORB features outside the mask; estimate motion with PnP + RANSAC.
4. Project LiDAR points into the image, drop those on moving objects, and align scans with ICP.
5. Optimize with a pose graph, bundle adjustment and loop closure.
6. Visualize with Rerun and score against ground truth (ATE / RPE).

## Tech stack

C++17 · CMake + presets · vcpkg · Eigen · OpenCV · yaml-cpp · GoogleTest · GitHub Actions · ASan / UBSan · evo (Python)  
*Planned:* Ceres, ONNX Runtime, Open3D, Rerun, Docker, ROS 2

## Build

**Requirements:** a C++17 compiler (Clang, GCC or MSVC), CMake ≥ 3.25, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with the `VCPKG_ROOT` environment variable set. On macOS, also `brew install pkg-config`.

```bash
git clone https://github.com/Adiaharoni55/realtime-visual-slam.git
cd realtime-visual-slam

cmake --preset debug             # configure; vcpkg installs dependencies on first run
cmake --build --preset debug     # build
ctest --test-dir build/debug --output-on-failure   # run unit tests
./build/debug/rtvm               # run the app
```

Use the `release` preset for optimized builds. Debug builds run with AddressSanitizer and UndefinedBehaviorSanitizer.

## Data

Download the TUM RGB-D sequence [`freiburg1_xyz`](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) (tgz) and unpack it into `data/`:

```bash
mkdir -p data && cd data
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
```

The results above also use `freiburg2_desk` and `freiburg1_desk`, from the same page and unpacked the same way. Use `configs/tum_fr1.yaml` for fr1 sequences and `configs/tum_fr2.yaml` for fr2 sequences.

The `data/` folder is not tracked by Git.

## Project structure

```
realtime-visual-slam/
├── include/rtvm/   # Public headers
├── src/            # Library implementation
├── apps/           # Programs: rtvm, rtvm_match_demo, rtvm_run_vo
├── tests/          # Unit tests and small test fixtures
├── configs/        # Camera parameters (YAML)
├── scripts/        # Python helpers (plotting, evaluation)
├── docs/           # Failure analysis and images
├── CMakeLists.txt
├── CMakePresets.json
└── vcpkg.json      # Dependencies
```

## References

- J. Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012 (TUM RGB-D dataset).

## License

MIT