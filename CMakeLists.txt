cmake_minimum_required(VERSION 3.29)
project(auldoCV)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
include_directories(include)

add_library(auldoCV
        src/gradient/compute_node.cpp
        src/gradient/compute_node_functions.cpp)

target_link_libraries(auldoCV ${OpenCV_LIBS})

add_executable(auldoCV_playground playground/playground.cpp)
target_link_libraries(auldoCV_playground auldoCV)