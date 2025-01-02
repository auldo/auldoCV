#pragma once
#include "activation.h"

class Neuron {
protected:
    explicit Neuron(ActivationFunction activation);
    ActivationFunction _activation;
public:
    std::shared_ptr<ComputeNode> _output_node;
    virtual std::string getType() = 0;
};
