cmake_minimum_required(VERSION 3.29)
project(auldoCV)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_library(auldoCV
        src/gradient/compute_node.cpp
        src/gradient/compute_node_functions.cpp
        src/dataset/read_cifar_10.cpp
        src/vision/base_kernel.cpp
        src/vision/simple_kernel.cpp
        src/neuron/fc_neuron.cpp
        src/neuron/activation.cpp
        src/layer/layer.cpp
        src/layer/fc_layer.cpp
        src/loss/mse_loss.cpp
        src/initialization/xavier.cpp
        src/optimizer/sgd_optimizer.cpp
        src/neuron/neuron.cpp
        src/loss/binary_cross_entropy_loss.cpp)
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
