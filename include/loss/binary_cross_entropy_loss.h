#pragma once

#include "loss/loss.h"
#include "layer/fc_layer.h"

class BinaryCrossEntropyLoss : public Loss {
public:
    explicit BinaryCrossEntropyLoss(PRECISE_NBR label, const PTR<ComputeNode>& node);
};