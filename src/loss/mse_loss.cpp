#include "loss/mse_loss.h"

MSELoss::MSELoss(PRECISE_NBR label, const std::shared_ptr<Layer>& finalLayer) {
    if(auto layer = std::dynamic_pointer_cast<FCLayer>(finalLayer)) {
        if(layer->_neurons.size() != 1)
            throw std::invalid_argument("mse currently only works for one output neuron in the final layer.");
        auto output{layer->_neurons.at(0)->_output_node};
        auto neg{COMPUTE_NODE_TIMES_CONST(output, -1)};
        auto sum{COMPUTE_NODE_PLUS_CONST(neg, label)};
        _output_node = COMPUTE_NODE_POW(sum, 2);
    } else throw std::runtime_error("mse loss currently only implemented for fc layer.");
}