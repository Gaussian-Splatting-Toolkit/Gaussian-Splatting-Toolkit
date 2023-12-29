# project
cmake_minimum_required(VERSION 3.15.0)
project(surface_distance)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(surface_distance src/main.cpp)
