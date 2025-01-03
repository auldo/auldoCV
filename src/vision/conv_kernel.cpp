#include "vision/conv_kernel.h"

#include <initialization/xavier.h>

ConvolutionalKernel::ConvolutionalKernel(PIXEL depth, PIXEL stride, PIXEL size, std::optional<PIXEL> padding) : BaseKernel(stride, size, padding), _depth(depth) {
    _weights = std::make_shared<BaseTensor<std::shared_ptr<ComputeNode>>>(Vector<INDEX_NBR>({size, size, _depth}));
    xavier(_size, _depth, _weights, _size * _size * _depth);
}