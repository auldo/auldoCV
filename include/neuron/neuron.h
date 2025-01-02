#pragma once
#include "activation.h"

class Neuron {
protected:
    explicit Neuron(ActivationFunction activation): _activation(activation) {};
    ActivationFunction _activation;
public:
    std::shared_ptr<ComputeNode> _output_node;
};
