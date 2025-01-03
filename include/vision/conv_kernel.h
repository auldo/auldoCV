#pragma once

#include "gradient/compute_node.h"
#include "vision/base_kernel.h"

class ConvolutionalKernel : public BaseKernel {
public:
    std::shared_ptr<Tensor<std::shared_ptr<ComputeNode>>> _weights;
    PIXEL _depth;
    ConvolutionalKernel(PIXEL _depth, PIXEL stride, PIXEL size, std::optional<PIXEL> padding = std::nullopt);
};