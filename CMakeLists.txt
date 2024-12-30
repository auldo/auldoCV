cmake_minimum_required(VERSION 3.29)
project(auldoCV)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_library(auldoCV
        src/gradient/compute_node.cpp
        src/gradient/compute_node_functions.cpp)
target_include_directories(auldoCV PUBLIC include)
target_link_libraries(auldoCV ${OpenCV_LIBS})

add_executable(auldoCV_playground playground/playground.cpp)
target_link_libraries(auldoCV_playground auldoCV)

find_package(doctest REQUIRED)
add_executable(auldoCV_tests test/tests.cpp)
target_include_directories(auldoCV_tests PUBLIC test/include)
target_compile_features(auldoCV_tests PRIVATE cxx_std_17)
target_link_libraries(auldoCV_tests PRIVATE doctest::doctest)
target_link_libraries(auldoCV_tests PRIVATE auldoCV)
