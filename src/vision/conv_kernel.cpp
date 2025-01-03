#include "vision/conv_kernel.h"

#include <initialization/xavier.h>

ConvolutionalKernel::ConvolutionalKernel(PIXEL depth, PIXEL stride, PIXEL size, std::optional<PIXEL> padding) : BaseKernel(stride, size, padding), _depth(depth) {
    _weights = std::make_shared<BaseTensor<std::shared_ptr<ComputeNode>>>(Vector<INDEX_NBR>({_size, _size, _depth}));
    for(INDEX_NBR r{0}; r < _size; ++r) {
        for(INDEX_NBR c{0}; c < _size; ++c) {
            for(INDEX_NBR d{0}; d < _depth; ++d) {
                _weights->at({r, c, d}) = COMPUTE_NODE(0);
            }
        }
    }
    xavier(_size, _depth, _weights, _size * _size * _depth);
}