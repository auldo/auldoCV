#include "loss/binary_cross_entropy_loss.h"

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(PRECISE_NBR label, const std::shared_ptr<Layer>& finalLayer) {
    if(auto layer = std::dynamic_pointer_cast<FCLayer>(finalLayer)) {
        if(layer->_neurons.size() != 1)
            throw std::invalid_argument("bce currently only works for one output neuron in the final layer.");

        auto truth{COMPUTE_NODE(label)};
        auto prediction{layer->_neurons.at(0)->_output_node};
        auto negPrediction{COMPUTE_NODE_TIMES_CONST(prediction, -1)};
        auto increasedNegPred{COMPUTE_NODE_PLUS_CONST(negPrediction, 1)};
        auto logIncrNegPred{COMPUTE_NODE_LOG_N(increasedNegPred)};
        auto negTruth{COMPUTE_NODE_TIMES_CONST(truth, -1)};
        auto increasedNegTruth{COMPUTE_NODE_PLUS_CONST(negTruth, 1)};
        auto rightSummand{COMPUTE_NODE_TIMES(increasedNegTruth, logIncrNegPred)};

        auto logPrediction{COMPUTE_NODE_LOG_N(prediction)};
        auto leftSummand{COMPUTE_NODE_TIMES(truth, logPrediction)};

        auto sum {COMPUTE_NODE_PLUS(leftSummand, rightSummand)};
        _output_node = COMPUTE_NODE_TIMES_CONST(sum, -1);
    } else throw std::runtime_error("bce loss currently only supported for fc layer.");
}