#pragma once

class Neuron {
protected:
    Neuron() = default;
public:
    std::shared_ptr<ComputeNode> _output_node;
};