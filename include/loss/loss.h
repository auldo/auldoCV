#pragma once

#include "gradient/compute_node.h"

class Loss {
protected:
    Loss(): _output_node(nullptr) {}
public:
    std::shared_ptr<ComputeNode> _output_node;
};