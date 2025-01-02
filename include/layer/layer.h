#pragma once
#include <neuron/neuron.h>

class Layer {
public:
    Vector<std::shared_ptr<Neuron>> _neurons;
    std::shared_ptr<Layer> _previous;
    explicit Layer(INDEX_NBR size, const std::shared_ptr<Layer>& previous = nullptr);
    USE_RETURN virtual Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const = 0;
};
