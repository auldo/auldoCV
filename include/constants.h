#pragma once

#include <memory>

#define PRECISE_NBR double
#define INDEX_NBR unsigned int

#define PTR std::shared_ptr
#define CREATE_PTR std::make_shared

#define PIXEL std::uint8_t
#define USE_RETURN [[nodiscard]]

#define COMPUTE_NODE(x) CREATE_PTR<ComputeNode>(x)
#define COMPUTE_NODE_PLUS(x, y) CREATE_PTR<ComputeNode>(OP_PLUS, std::vector({x, y}))
#define COMPUTE_NODE_POW(x, y) CREATE_PTR<ComputeNode>(OP_POW_CONST_EXPONENT, std::vector({x}), std::vector<PRECISE_NBR>({y}))
#define COMPUTE_NODE_TIMES(x, y) CREATE_PTR<ComputeNode>(OP_TIMES, std::vector({x, y}))
#define COMPUTE_NODE_RELU(x) CREATE_PTR<ComputeNode>(FN_RELU, std::vector({x}))
#define COMPUTE_NODE_TIME_CONST(x, y) CREATE_PTR<ComputeNode>(OP_TIMES_CONST, std::vector({x}), std::vector<PRECISE_NBR>({y}))