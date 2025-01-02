#pragma once

#include "gradient/compute_node.h"

class Loss {
public:
    std::shared_ptr<ComputeNode> _output_node;
    Loss(): _output_node(nullptr) {}
};

enum LossFunction {
    MSE
};