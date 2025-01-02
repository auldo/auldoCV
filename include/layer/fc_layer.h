#pragma once
#include "layer.h"

class FCLayer : public Layer {
public:
    std::optional<Vector<std::shared_ptr<ComputeNode>>> _inputs;
    explicit FCLayer(INDEX_NBR size, ActivationFunction activation, INDEX_NBR inputSize);
    explicit FCLayer(INDEX_NBR size, ActivationFunction activation, const std::shared_ptr<Layer>& layer);
    USE_RETURN Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const override;
};