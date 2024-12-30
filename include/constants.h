#pragma once

#include <memory>

#define PRECISE_NBR double
#define INDEX_NBR unsigned int

#define PTR std::shared_ptr
#define CREATE_PTR std::make_shared

#define PIXEL std::uint8_t
#define USE_RETURN [[nodiscard]]

#define COMPUTE_NODE(x) CREATE_PTR<ComputeNode>(x)
#define COMPUTE_NODE_PLUS(x, y) CREATE_PTR<ComputeNode>(OP_PLUS, Vector({x, y}))
#define COMPUTE_NODE_PLUS_CONST(x, y) CREATE_PTR<ComputeNode>(OP_PLUS, Vector({x}), Vector<PRECISE_NBR>({y}))
#define COMPUTE_NODE_TIMES(x, y) CREATE_PTR<ComputeNode>(OP_TIMES, Vector({x, y}))
#define COMPUTE_NODE_TIMES_CONST(x, y) CREATE_PTR<ComputeNode>(OP_TIMES_CONST, Vector({x}), Vector<PRECISE_NBR>({y}))
#define COMPUTE_NODE_DIV(x, y) CREATE_PTR<ComputeNode>(OP_DIV, Vector({x, y}))
#define COMPUTE_NODE_DIV_CONST(x, y) CREATE_PTR<ComputeNode>(OP_DIV_CONST_DIVIDEND, Vector({x}), Vector<PRECISE_NBR>({y}))
#define COMPUTE_NODE_POW(x, y) CREATE_PTR<ComputeNode>(OP_POW_CONST_EXPONENT, Vector({x}), Vector<PRECISE_NBR>({y}))
#define COMPUTE_NODE_E(x) CREATE_PTR<ComputeNode>(FN_EXP, Vector({x}))
#define COMPUTE_NODE_RELU(x) CREATE_PTR<ComputeNode>(FN_RELU, Vector({x}))
#define COMPUTE_NODE_SIGMOID(x) CREATE_PTR<ComputeNode>(FN_SIGMOID, Vector({x}))
#define COMPUTE_NODE_LOG_N(x) CREATE_PTR<ComputeNode>(FN_LOG_N, Vector({x}))