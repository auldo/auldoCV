# Global settings
cmake_minimum_required(VERSION 3.29)
project(auldoCV)
set(CMAKE_CXX_STANDARD 20)

# auldoCV library
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
        src/loss/binary_cross_entropy_loss.cpp
        src/vision/conv_kernel.cpp
        src/layer/conv_layer.cpp
        src/loss/cross_entropy_loss.cpp)
find_package(OpenCV REQUIRED)
target_include_directories(auldoCV PUBLIC ${OPENCV_INCLUDE_DIRS})
target_include_directories(auldoCV PUBLIC include)
target_link_libraries(auldoCV ${OpenCV_LIBS})

# playground executable
# comment the following two lines if you don't have a playground as this is not pushed onto the repo
add_executable(auldoCV_playground playground/playground.cpp)
target_link_libraries(auldoCV_playground auldoCV)

# test executable
find_package(doctest REQUIRED)
add_executable(auldoCV_tests test/tests.cpp)
target_include_directories(auldoCV_tests PUBLIC test/include)
target_compile_features(auldoCV_tests PRIVATE cxx_std_17)
target_link_libraries(auldoCV_tests PRIVATE doctest::doctest)
target_link_libraries(auldoCV_tests PRIVATE auldoCV)

# example executables
add_executable(auldoCV_example_2d_function_approximation examples/2d_function_approximation/main.cpp)
target_link_libraries(auldoCV_example_2d_function_approximation PRIVATE auldoCV)
