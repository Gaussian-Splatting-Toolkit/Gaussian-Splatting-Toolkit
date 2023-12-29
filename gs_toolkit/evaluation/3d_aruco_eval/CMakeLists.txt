# project
cmake_minimum_required(VERSION 3.15.0)
project(3d_aruco_eval)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

################### dependencies ###################
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

############## Source and Testing ##################
include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(calibrate_camera src/calibrate_camera.cpp)
target_link_libraries(calibrate_camera ${OpenCV_LIBS})

add_executable(detect_markers src/detect_markers.cpp)
target_link_libraries(detect_markers ${OpenCV_LIBS})

add_executable(generate src/generate.cpp)
target_link_libraries(generate ${OpenCV_LIBS})
