#pragma once

#include "data/tensor.h"
#include "loss/loss.h"
#include "layer/layer.h"
#include "optimizer/optimizer.h"

/// For one epoch, we consider EACH sample.
/// Take sample, let it flow through the network update weights with factor 1/n (which is actually just scaling the lr).
/// Take next sample, ...
class SgdOptimizer : public Optimizer  {
    INDEX_NBR _epochs;
public:
    SgdOptimizer(const std::shared_ptr<Layer>& finalLayer, INDEX_NBR epochs, LossFunction loss, const std::shared_ptr<Tensor<PRECISE_NBR>>& truth, const std::shared_ptr<Tensor<PRECISE_NBR>>& inputs);
    void optimize(PRECISE_NBR learningRate) override;
};