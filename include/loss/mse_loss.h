#pragma once

#include "loss/loss.h"
#include "layer/fc_layer.h"

class MSELoss : public Loss {
public:
    explicit MSELoss(PRECISE_NBR label, const PTR<ComputeNode>& node);
};