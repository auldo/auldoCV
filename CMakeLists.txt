cmake_minimum_required(VERSION 3.29)
project(auldo_machine_learning)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(linear_regression src/playground.cpp
        src/read/tab_reparated.cpp
        src/read/util.cpp
        src/read/read_image.cpp
        src/write/write_image.cpp)

target_link_libraries(linear_regression ${OpenCV_LIBS})