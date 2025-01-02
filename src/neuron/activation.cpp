#include "neuron/activation.h"

std::shared_ptr<ComputeNode> applyActivationFunction(const ActivationFunction& activation, const std::shared_ptr<ComputeNode>& node) {
    switch(activation) {
        case LINEAR:
            return node;
        case SIGMOID:
            return COMPUTE_NODE_SIGMOID(node);
        case RELU:
            return COMPUTE_NODE_RELU(node);
        default:
            throw std::runtime_error("unsupported activation function");
    }
}