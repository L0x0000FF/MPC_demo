cmake_minimum_required(VERSION 3.9)

project(traj)


enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_FLAGS "-std=c++17")

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

add_executable(main main.cpp Trajectory.cpp)
