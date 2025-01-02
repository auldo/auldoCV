#include "initialization/xavier.h"

/// n being the number of input neurons / inputs for first layer.
void xavier(Vector<std::shared_ptr<ComputeNode>>& data, INDEX_NBR n) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, sqrt(1. / TO_PRECISE_NBR(n)));
    for(auto& node : data) {
        PRECISE_NBR generated{dist(e2)};
        node->setScalarValue(generated);
    }
}

void xavier(std::shared_ptr<ComputeNode>& node, INDEX_NBR n) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, sqrt(1. / TO_PRECISE_NBR(n)));
    PRECISE_NBR generated{dist(e2)};
    node->setScalarValue(generated);
}