#pragma once

#include "data/vector.h"
#include "neuron.h"

class ComputeNode;

class FCNeuron : public Neuron {
    explicit FCNeuron(INDEX_NBR size);
public:
    Vector<std::shared_ptr<ComputeNode>> _weights;
    std::shared_ptr<ComputeNode> _bias;
    explicit FCNeuron(const Vector<std::shared_ptr<ComputeNode>>& inputs);
};
