#pragma once

/// The possible types of computational graph nodes.
/// For those types forward and backward passes (i.e., (partial) derivatives) have been implemented.
enum ComputeNodeType {
    OP_PLUS, OP_PLUS_CONST,
    OP_TIMES, OP_TIMES_CONST,
    OP_DIV, OP_DIV_CONST_DIVIDEND,
    OP_POW_CONST_EXPONENT,
    FN_EXP,
    FN_RELU,
    FN_LOG_N,
    FN_SIGMOID,
    SCALAR
};