#include "loss/cross_entropy_loss.h"

CrossEntropyLoss::CrossEntropyLoss(PRECISE_NBR label, const Vector<PTR<ComputeNode>>& nodes) {
    const INDEX_NBR classCount{nodes.size()};

    std::shared_ptr<ComputeNode> classSum{nullptr};
    for(auto k{0}; k < classCount; ++k) {
        auto e{COMPUTE_NODE_E(nodes.at(k))};
        classSum = classSum == nullptr ? e : COMPUTE_NODE_PLUS(classSum, e);
    }

    auto scoreAtActual{nodes.at(TO_INDEX_NBR(label))};
    auto scoreAtActualE{COMPUTE_NODE_E(scoreAtActual)};

    auto div{COMPUTE_NODE_DIV(scoreAtActualE, classSum)};
    auto log{COMPUTE_NODE_LOG_N(div)};
    _output_node = COMPUTE_NODE_TIMES_CONST(log, -1);
}
