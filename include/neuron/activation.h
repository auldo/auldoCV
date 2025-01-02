#pragma once

#include "gradient/compute_node.h"

enum ActivationFunction {
    LINEAR /*i.e., none*/,
    SIGMOID,
    RELU
};

std::shared_ptr<ComputeNode> applyActivationFunction(const ActivationFunction& activation, const std::shared_ptr<ComputeNode>& node);