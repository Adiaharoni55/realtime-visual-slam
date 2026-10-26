# Oldest CMake version this file works with (3.25 is needed for presets later)
cmake_minimum_required(VERSION 3.25)

# Project name, version, and language (CXX = C++)
project(rtvm VERSION 0.1.0 LANGUAGES CXX)

# ---- Dependencies (installed by vcpkg from vcpkg.json) ----
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
# DBoW2's own CMake config is outdated (no target, release library only), so define
# a proper imported target: headers + the release and debug builds of the library.
set(RTVM_VCPKG_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
find_path(DBOW2_INCLUDE_DIR DBoW2/TemplatedVocabulary.h
          PATHS "${RTVM_VCPKG_DIR}/include" NO_DEFAULT_PATH REQUIRED)
find_library(DBOW2_LIBRARY_RELEASE DBoW2
             PATHS "${RTVM_VCPKG_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(DBOW2_LIBRARY_DEBUG DBoW2
             PATHS "${RTVM_VCPKG_DIR}/debug/lib" NO_DEFAULT_PATH)

add_library(DBoW2::DBoW2 UNKNOWN IMPORTED)
set_target_properties(DBoW2::DBoW2 PROPERTIES
    IMPORTED_LOCATION "${DBOW2_LIBRARY_RELEASE}"
    INTERFACE_INCLUDE_DIRECTORIES "${DBOW2_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${OpenCV_LIBS}")
if(DBOW2_LIBRARY_DEBUG)
    set_target_properties(DBoW2::DBoW2 PROPERTIES
        IMPORTED_LOCATION_DEBUG "${DBOW2_LIBRARY_DEBUG}")
endif()

# Use C++17 for the whole project, and fail if the compiler can't do it
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # plain standard C++, no compiler-specific extras

# Write compile_commands.json so VS Code understands the project
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ceres -> glog -> gflags puts the same static library on the link line more than once.
# That's harmless, but Apple's linker (Xcode 15+) warns about it, so silence that warning.
if(APPLE)
    include(CheckLinkerFlag)
    check_linker_flag(CXX "-Wl,-no_warn_duplicate_libraries" RTVM_HAS_NO_WARN_DUP_LIBS)
    if(RTVM_HAS_NO_WARN_DUP_LIBS)
        add_link_options(-Wl,-no_warn_duplicate_libraries)
    endif()
endif()

# ---- Library: the reusable SLAM code ----
add_library(rtvm_core
    src/version.cpp
    src/camera.cpp
    src/pose.cpp
    src/tum_dataset.cpp
    src/depth_source.cpp
    src/features.cpp
    src/motion.cpp
    src/visual_odometry.cpp
    src/pose_graph.cpp
    src/loop_detector.cpp
    src/vocabulary.cpp
)

# find the version
target_compile_definitions(rtvm_core PRIVATE RTVM_VERSION="${PROJECT_VERSION}")

# Anyone using rtvm_core can #include "rtvm/..." from the include/ folder
target_include_directories(rtvm_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# ---- Links ----
# PUBLIC: our headers use these types, so users of rtvm_core need them too.
# PRIVATE: used only inside our .cpp files.
target_link_libraries(rtvm_core
    PUBLIC  Eigen3::Eigen ${OpenCV_LIBS} Ceres::ceres DBoW2::DBoW2
    PRIVATE yaml-cpp::yaml-cpp
)

# ---- App: the rtvm command-line program ----
add_executable(rtvm
    apps/main.cpp
)

# ---- Demo: ORB matching between two TUM frames (writes matches.png) ----
add_executable(rtvm_match_demo apps/match_demo.cpp)
target_link_libraries(rtvm_match_demo PRIVATE rtvm_core rtvm_warnings rtvm_sanitizers)

# ---- App: run visual odometry over a TUM sequence (writes output/trajectory.txt) ----
add_executable(rtvm_run_vo apps/run_vo.cpp)
target_link_libraries(rtvm_run_vo PRIVATE rtvm_core rtvm_warnings rtvm_sanitizers)

# ---- Tool: train an ORB vocabulary from TUM sequences (writes e.g. models/orb_vocab.yml.gz) ----
add_executable(rtvm_train_vocab apps/train_vocab.cpp)
target_link_libraries(rtvm_train_vocab PRIVATE rtvm_core rtvm_warnings rtvm_sanitizers)

# ---- Tool: convert a vocabulary between formats (e.g. YAML -> fast-loading .bin) ----
add_executable(rtvm_convert_vocab apps/convert_vocab.cpp)
target_link_libraries(rtvm_convert_vocab PRIVATE rtvm_core rtvm_warnings rtvm_sanitizers)

# The app uses the library (and inherits its PUBLIC include folder)
target_link_libraries(rtvm PRIVATE rtvm_core)

# ---- Compiler warnings ----
# A small "interface" target that holds only settings, no code.
# Any target that links it gets these warning flags.
add_library(rtvm_warnings INTERFACE)
target_compile_options(rtvm_warnings INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(rtvm_core PRIVATE rtvm_warnings)
target_link_libraries(rtvm      PRIVATE rtvm_warnings)

# ---- Sanitizers (Debug builds only) ----
# Catch memory bugs (ASan) and undefined behavior (UBSan) at runtime.
option(RTVM_SANITIZERS "Enable ASan + UBSan in Debug builds" ON)

add_library(rtvm_sanitizers INTERFACE)
if(RTVM_SANITIZERS AND NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(rtvm_sanitizers INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer   # gives readable stack traces in error reports
    )
    target_link_options(rtvm_sanitizers INTERFACE
        -fsanitize=address,undefined
    )
endif()

target_link_libraries(rtvm_core PRIVATE rtvm_sanitizers)
target_link_libraries(rtvm      PRIVATE rtvm_sanitizers)

# ---- Unit tests ----
option(RTVM_BUILD_TESTS "Build unit tests" ON)

if(RTVM_BUILD_TESTS)
    enable_testing()

    # Download GoogleTest automatically (first configure only)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # needed on Windows (MSVC)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()