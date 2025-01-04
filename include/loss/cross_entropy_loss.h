#pragma once

#include "loss/loss.h"
#include "layer/fc_layer.h"
#include "data/tensor.h"

class CrossEntropyLoss : public Loss {
public:
    explicit CrossEntropyLoss(PIXEL label, const std::shared_ptr<Layer>& finalLayer);
};