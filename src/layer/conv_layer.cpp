#include "layer/conv_layer.h"

ConvolutionalLayer::ConvolutionalLayer(PIXEL filterSize, PIXEL filterStride, INDEX_NBR kernelCount, INDEX_NBR inputHeight, INDEX_NBR inputWidth, INDEX_NBR inputDepth) : _inputHeight(inputHeight), _inputWidth(inputWidth), _inputDepth(inputDepth), _kernels(kernelCount) {
    _input = std::make_shared<BaseTensor<std::shared_ptr<ComputeNode>>>(Vector({inputHeight, inputWidth, inputDepth}));
    for(INDEX_NBR r{0}; r < inputHeight; ++r) {
        for(INDEX_NBR c{0}; c < inputWidth; ++c) {
            for(INDEX_NBR ch{0}; ch < inputDepth; ++ch) {
                _input->at({r, c, ch}) = COMPUTE_NODE(0);
            }
        }
    }

    for(INDEX_NBR k{0}; k < _kernels.size(); ++k)
        _kernels.at(k) = std::make_shared<ConvolutionalKernel>(inputDepth, filterStride, filterSize);


    INDEX_NBR newHeight = _kernels.at(0)->calculateOutputDimension(inputHeight);
    INDEX_NBR newWidth = _kernels.at(0)->calculateOutputDimension(inputWidth);
    _output = std::make_shared<BaseTensor<std::shared_ptr<ComputeNode>>>(Vector({newHeight, newWidth, kernelCount}));
    //Iterate over each kernel of this layer.
    //Each kernel represents an output layer.
    //Each kernel runs over all input layers and reduces them in 1 single layer.

    for(INDEX_NBR k{0}; k < _kernels.size(); ++k) {

        //Run over every pixel of new layer
        for(INDEX_NBR r{0}; r < newHeight; ++r) {
            for(INDEX_NBR c{0}; c < newWidth; ++c) {
                auto pixel{COMPUTE_NODE(0)};

                INDEX_NBR startX{r * filterStride};
                INDEX_NBR startY{c * filterStride};

                for(INDEX_NBR x{0}; x < filterSize; ++x) {
                    for(INDEX_NBR y{0}; y < filterSize; ++y) {
                        for(INDEX_NBR ch{0}; ch < inputDepth; ++ch) {
                            auto tmp{COMPUTE_NODE_TIMES(_input->at({x + startX, y + startY, ch}), _kernels.at(k)->_weights->at({x, y, ch}))};
                            pixel = COMPUTE_NODE_PLUS(pixel, tmp);
                        }
                    }
                }

                _output->at({r, c, k}) = pixel;
            }
        }
    }
}

void ConvolutionalLayer::setInputs(const std::shared_ptr<Tensor<PIXEL>> &input) const {
    if(input->shapeSize(0) != _inputHeight)
        throw std::runtime_error("bad img height");
    if(input->shapeSize(1) != _inputWidth)
        throw std::runtime_error("bad img width");
    if(input->shapeSize(2) != _inputDepth)
        throw std::runtime_error("bad img depth");
    for(INDEX_NBR r{0}; r < _inputHeight; ++r) {
        for(INDEX_NBR c{0}; c < _inputWidth; ++c) {
            for(INDEX_NBR ch{0}; ch < _inputDepth; ++ch)
                _input->at({r, c, ch})->setScalarValue(TO_PRECISE_NBR(input->at({r, c, ch})));
        }
    }
}


Vector<std::shared_ptr<ComputeNode>> ConvolutionalLayer::getComputeNodes() const {
    throw std::runtime_error("Not implemented");
}
