#pragma once

#include "constants.h"
#include "layer/layer.h"
#include "loss/loss.h"
#include "layer/fc_layer.h"
#include "layer/conv_layer.h"
#include "neuron/fc_neuron.h"
#include "data/tensor.h"

class Optimizer {
protected:
    LossFunction _lossFunction;
    std::shared_ptr<Layer> _layer;
    std::shared_ptr<Tensor<PRECISE_NBR>> _truth;
    std::shared_ptr<Tensor<PRECISE_NBR>> _inputs;
    Optimizer(const std::shared_ptr<Layer>& layer, LossFunction loss, const std::shared_ptr<Tensor<PRECISE_NBR>>& truth, const std::shared_ptr<Tensor<PRECISE_NBR>>& inputs);
    void updateWeights(PRECISE_NBR learningRate) const;
    void updateAverageWeights(PRECISE_NBR learningRate) const;
    void updateAverageCloneWeights(PRECISE_NBR learningRate) const;
    void rescaleGradientStorages(INDEX_NBR size) const;
    void setGradientStorage(INDEX_NBR idx) const;
public:
    virtual ~Optimizer() = default;
    virtual void optimize(PRECISE_NBR learningRate) = 0;
};