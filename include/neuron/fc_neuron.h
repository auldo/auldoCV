#pragma once

#include <layer/layer.h>

#include "data/vector.h"
#include "neuron.h"

class ComputeNode;

class FCNeuron : public Neuron {
    explicit FCNeuron(ActivationFunction activation, INDEX_NBR size);
public:
    Vector<std::shared_ptr<ComputeNode>> _weights;
    std::shared_ptr<ComputeNode> _bias;
    explicit FCNeuron(ActivationFunction activation, const Vector<std::shared_ptr<ComputeNode>>& inputs);
    explicit FCNeuron(ActivationFunction activation, const std::shared_ptr<Layer>& layer);
};
