#pragma once
#include <neuron/neuron.h>

class Layer {
public:
    virtual ~Layer() = default;
    std::shared_ptr<Layer> _previous;
    explicit Layer(const std::shared_ptr<Layer>& previous = nullptr);

    /// Those should be renamed to outputs.
    /// For conv layer those are the linearized image.
    /// For fc layer those are the neuron output nodes.
    USE_RETURN virtual Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const = 0;
};
