#include "layer/conv_layer.h"

void ConvolutionalLayer::buildOutputNode(ActivationFunction activation, INDEX_NBR newHeight, INDEX_NBR newWidth, INDEX_NBR filterStride, INDEX_NBR filterSize) {
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
                        for(INDEX_NBR ch{0}; ch < _inputDepth; ++ch) {
                            auto tmp{COMPUTE_NODE_TIMES(_input->at({x + startX, y + startY, ch}), _kernels.at(k)->_weights->at({x, y, ch}))};
                            pixel = COMPUTE_NODE_PLUS(pixel, tmp);
                        }
                    }
                }

                _output->at({r, c, k}) = applyActivationFunction(activation, pixel);
            }
        }
    }
}

ConvolutionalLayer::ConvolutionalLayer(ActivationFunction activation, const std::shared_ptr<ConvolutionalLayer>& previous, PIXEL filterStride, PIXEL filterSize, INDEX_NBR kernelCount) : Layer(previous), _kernels(kernelCount) {
    _inputHeight = previous->_output->shapeSize(0);
    _inputWidth = previous->_output->shapeSize(1);
    _inputDepth = previous->_output->shapeSize(2);

    _input = previous->_output;

    for(INDEX_NBR k{0}; k < _kernels.size(); ++k)
        _kernels.at(k) = std::make_shared<ConvolutionalKernel>(_inputDepth, filterStride, filterSize);

    INDEX_NBR newHeight = _kernels.at(0)->calculateOutputDimension(_inputHeight);
    INDEX_NBR newWidth = _kernels.at(0)->calculateOutputDimension(_inputWidth);

    _output = std::make_shared<BaseTensor<std::shared_ptr<ComputeNode>>>(Vector({newHeight, newWidth, kernelCount}));
    buildOutputNode(activation, newHeight, newWidth, filterStride, filterSize);
}

ConvolutionalLayer::ConvolutionalLayer(ActivationFunction activation, PIXEL filterSize, PIXEL filterStride, INDEX_NBR kernelCount, INDEX_NBR inputHeight, INDEX_NBR inputWidth, INDEX_NBR inputDepth) : _inputHeight(inputHeight), _inputWidth(inputWidth), _inputDepth(inputDepth), _kernels(kernelCount) {
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
    buildOutputNode(activation, newHeight, newWidth, filterStride, filterSize);
}

void ConvolutionalLayer::setInputs(const std::shared_ptr<Tensor<PRECISE_NBR>> &input) const {
    if(input->shapeSize(0) != _inputHeight)
        throw std::runtime_error("bad img height");
    if(input->shapeSize(1) != _inputWidth)
        throw std::runtime_error("bad img width");
    if(input->shapeSize(2) != _inputDepth)
        throw std::runtime_error("bad img depth");
    for(INDEX_NBR r{0}; r < _inputHeight; ++r) {
        for(INDEX_NBR c{0}; c < _inputWidth; ++c) {
            for(INDEX_NBR ch{0}; ch < _inputDepth; ++ch)
                _input->at({r, c, ch})->setScalarValue(input->at({r, c, ch}));
        }
    }
}


Vector<std::shared_ptr<ComputeNode>> ConvolutionalLayer::getComputeNodes() const {
    Vector<std::shared_ptr<ComputeNode>> nodes(_output->shapeSize(0) * _output->shapeSize(1) * _output->shapeSize(2));
    auto vectorIdx{0};
    for(INDEX_NBR r{0}; r < _output->shapeSize(0); ++r) {
        for(INDEX_NBR c{0}; c < _output->shapeSize(1); ++c) {
            for(INDEX_NBR ch{0}; ch < _output->shapeSize(2); ++ch) {
                nodes.at(vectorIdx) = _output->at({r, c, ch});
                ++vectorIdx;
            }
        }
    }
    return nodes;
}
