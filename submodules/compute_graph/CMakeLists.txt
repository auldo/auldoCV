cmake_minimum_required(VERSION 3.29)
project(auldo_cv_compute_graph)

add_library(auldo_cv_compute_graph compute_node.c)
target_include_directories(auldo_cv_compute_graph PUBLIC ./)

# Playground
add_executable(auldo_cv_compute_graph_playground playground/playground.c)
target_link_libraries(auldo_cv_compute_graph_playground auldo_cv_compute_graph)

# Unity library
add_library(unity_lib tests/unity/unity.c)
target_include_directories(unity_lib PUBLIC tests/unity)

# Unity tests
add_executable(auldo_cv_compute_graph_tests tests/tests.c)
target_link_libraries(auldo_cv_compute_graph_tests auldo_cv_compute_graph unity_lib)