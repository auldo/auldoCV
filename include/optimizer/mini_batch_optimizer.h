#pragma once

#include <random>
#include "data/tensor.h"
#include "loss/binary_cross_entropy_loss.h"
#include "loss/cross_entropy_loss.h"
#include "loss/mse_loss.h"
#include "layer/conv_layer.h"
#include "layer/fc_layer.h"
#include "optimizer/optimizer.h"
#include "neuron/fc_neuron.h"

class MiniBatchOptimizer : public Optimizer  {
    INDEX_NBR _iterations;
    INDEX_NBR _batchSize;
public:
    USE_RETURN Vector<INDEX_NBR> selectMiniBatch() const;
    MiniBatchOptimizer(const std::shared_ptr<Layer>& finalLayer, INDEX_NBR iterations, INDEX_NBR miniBatchSize, LossFunction loss, const std::shared_ptr<Tensor<PRECISE_NBR>>& truth, const std::shared_ptr<Tensor<PRECISE_NBR>>& inputs);
    void optimize(PRECISE_NBR learningRate) override;
};