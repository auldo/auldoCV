#pragma once
#include <neuron/neuron.h>

class Layer {
public:
    virtual ~Layer() = default;
    std::shared_ptr<Layer> _previous;
    explicit Layer(const std::shared_ptr<Layer>& previous = nullptr);
    USE_RETURN virtual Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const = 0;
};
