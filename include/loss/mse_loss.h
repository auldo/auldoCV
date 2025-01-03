#pragma once

#include "loss/loss.h"
#include "layer/fc_layer.h"

class MSELoss : public Loss {
public:
    explicit MSELoss(PRECISE_NBR label, const std::shared_ptr<Layer>& finalLayer);
};