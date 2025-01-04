#pragma once

#include "loss/loss.h"
#include "data/tensor.h"

class CrossEntropyLoss : public Loss {
public:
    explicit CrossEntropyLoss(PRECISE_NBR label, const Vector<PTR<ComputeNode>>& nodes);
};