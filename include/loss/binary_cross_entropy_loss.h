#pragma once

#include "loss/loss.h"
#include "layer/fc_layer.h"

class BinaryCrossEntropyLoss : public Loss {
public:
    explicit BinaryCrossEntropyLoss(PRECISE_NBR label, const std::shared_ptr<Layer>& finalLayer);
};