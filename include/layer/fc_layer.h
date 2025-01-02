#pragma once
#include "layer.h"

class FCLayer : public Layer {
public:
    explicit FCLayer(INDEX_NBR size, ActivationFunction activation, const Vector<std::shared_ptr<ComputeNode>>& inputs);
    explicit FCLayer(INDEX_NBR size, ActivationFunction activation, const std::shared_ptr<Layer>& layer);
    USE_RETURN Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const override;
};