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
    test_dbow2.cpp
    test_loop_detector.cpp
    test_vocabulary.cpp
    test_bundle_adjustment.cpp
    test_segmentation.cpp
    test_visual_odometry.cpp
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

# Windows finds DLLs next to the program; this must run before the tests are discovered.
if(WIN32)
    add_custom_command(TARGET rtvm_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:rtvm_tests> $<TARGET_FILE_DIR:rtvm_tests>
        COMMAND_EXPAND_LISTS)
endif()

# Register each TEST() with CTest, CMake's test runner
include(GoogleTest)
gtest_discover_tests(rtvm_tests)