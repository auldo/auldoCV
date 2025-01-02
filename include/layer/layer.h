#pragma once
#include <neuron/neuron.h>

class Layer {
public:
    Vector<std::shared_ptr<Neuron>> _neurons;
    explicit Layer(INDEX_NBR size);
    virtual Vector<std::shared_ptr<ComputeNode>> getComputeNodes() const = 0;
};
