# Test program: all test files go in this list
add_executable(rtvm_tests
    test_version.cpp
    test_camera.cpp
    test_pose.cpp
    test_tum_dataset.cpp
    test_depth_source.cpp
    test_features.cpp
    test_motion.cpp
    test_ceres.cpp
    test_pose_graph.cpp
)

# Tests use our library + GoogleTest (gtest_main provides main())
target_link_libraries(rtvm_tests PRIVATE
    rtvm_core
    GTest::gtest_main
    rtvm_warnings
    rtvm_sanitizers
)

# Give tests the absolute path of the configs/ folder
target_compile_definitions(rtvm_tests PRIVATE
    RTVM_CONFIG_DIR="${PROJECT_SOURCE_DIR}/configs"
    RTVM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# Register each TEST() with CTest, CMake's test runner
include(GoogleTest)
gtest_discover_tests(rtvm_tests)