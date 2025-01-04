#include "loss/cross_entropy_loss.h"

CrossEntropyLoss::CrossEntropyLoss(PIXEL label, const std::shared_ptr<Layer>& finalLayer) {
    if(auto layer = std::dynamic_pointer_cast<FCLayer>(finalLayer)) {
        const INDEX_NBR classCount{layer->_neurons.size()};

        std::shared_ptr<ComputeNode> classSum{nullptr};
        for(auto k{0}; k < classCount; ++k) {
            auto e{COMPUTE_NODE_E(layer->_neurons.at(k)->_output_node)};
            classSum = classSum == nullptr ? e : COMPUTE_NODE_PLUS(classSum, e);
        }

        auto scoreAtActual{layer->_neurons.at(TO_INDEX_NBR(label))->_output_node};
        auto scoreAtActualE{COMPUTE_NODE_E(scoreAtActual)};

        auto div{COMPUTE_NODE_DIV(scoreAtActualE, classSum)};
        auto log{COMPUTE_NODE_LOG_N(div)};
        _output_node = COMPUTE_NODE_TIMES_CONST(log, -1);

    } else throw std::runtime_error("final layer must be fc layer.");
}
