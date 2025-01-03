#include "neuron/fc_neuron.h"

FCNeuron::FCNeuron(ActivationFunction activation, INDEX_NBR size): Neuron(activation), _weights(size) {
    for(INDEX_NBR i{0}; i < size; ++i)
        _weights.at(i) = COMPUTE_NODE(0);
    xavier(_weights, size);
    _bias = COMPUTE_NODE(0);
    xavier(_bias, size);
}

FCNeuron::FCNeuron(ActivationFunction activation, const Vector<std::shared_ptr<ComputeNode>>& inputs): FCNeuron(activation, inputs.size()) {
    Vector<std::shared_ptr<ComputeNode>> products(inputs.size());
    for(INDEX_NBR i{0}; i < inputs.size(); ++i)
        products.at(i) = COMPUTE_NODE_TIMES(inputs.at(i), _weights.at(i));
    auto sum{COMPUTE_NODE(0)};
    for(auto i{0}; i < products.size(); ++i)
        sum = COMPUTE_NODE_PLUS(sum, products.at(i));
    auto biasAdded{COMPUTE_NODE_PLUS(sum, _bias)};
    _output_node = applyActivationFunction(activation, biasAdded);
}

std::string FCNeuron::getType() {
    return "fc_neuron";
}


FCNeuron::FCNeuron(ActivationFunction activation, const std::shared_ptr<Layer>& layer): FCNeuron(activation, layer->getComputeNodes()) {}