#pragma once

#include "layer/fc_layer.h"
#include "neuron/fc_neuron.h"

FCLayer::FCLayer(INDEX_NBR size, ActivationFunction activation, INDEX_NBR inputSize): Layer(size), _inputs(inputSize) {
    for(auto i{0}; i < inputSize; ++i)
        _inputs->at(i) = COMPUTE_NODE(0);
    for(INDEX_NBR i = 0; i < size; ++i)
        _neurons.at(i) = std::make_shared<FCNeuron>(activation, _inputs.value());
}

FCLayer::FCLayer(INDEX_NBR size, ActivationFunction activation, const std::shared_ptr<Layer>& layer): Layer(size, layer) {
    for(INDEX_NBR i = 0; i < size; ++i)
        _neurons.at(i) = std::make_shared<FCNeuron>(activation, layer);
}

Vector<std::shared_ptr<ComputeNode>> FCLayer::getComputeNodes() const {
    Vector<std::shared_ptr<ComputeNode>> result(_neurons.size());
    for(INDEX_NBR i{0}; i < result.size(); ++i)
        result.at(i) = _neurons.at(i)->_output_node;
    return result;
}

