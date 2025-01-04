#pragma once

#include <data/tensor.h>
#include <vision/conv_kernel.h>

#include "layer/layer.h"

class ConvolutionalLayer : public Layer {
public:
    std::shared_ptr<Tensor<std::shared_ptr<ComputeNode>>> _input;
    std::shared_ptr<Tensor<std::shared_ptr<ComputeNode>>> _output;
    Vector<std::shared_ptr<ConvolutionalKernel>> _kernels;

    INDEX_NBR _inputHeight, _inputWidth, _inputDepth;

    /// When creating first conv layer use this constructor.
    ConvolutionalLayer(ActivationFunction activation, PIXEL filterSize, PIXEL filterStride, INDEX_NBR kernelCount, INDEX_NBR inputHeight, INDEX_NBR inputWidth, INDEX_NBR inputDepth);
    explicit ConvolutionalLayer(ActivationFunction activation, const std::shared_ptr<ConvolutionalLayer>& previous, PIXEL filterStride, PIXEL filterSize, INDEX_NBR kernelCount);
    void setInputs(const std::shared_ptr<Tensor<PRECISE_NBR>>& input) const;
    USE_RETURN Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const override;
    void buildOutputNode(ActivationFunction activation, INDEX_NBR newHeight, INDEX_NBR newWidth, INDEX_NBR filterStride, INDEX_NBR filterSize);
};