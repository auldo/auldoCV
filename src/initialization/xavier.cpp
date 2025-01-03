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

void xavier(PIXEL& size, PIXEL& depth, std::shared_ptr<Tensor<std::shared_ptr<ComputeNode>>>& data, INDEX_NBR n) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, sqrt(1. / TO_PRECISE_NBR(n)));
    for(PIXEL r{0}; r < size; ++r) {
        for(PIXEL c{0}; c < size; ++c) {
            for(PIXEL d{0}; d < depth; ++d) {
                PRECISE_NBR generated{dist(e2)};
                data->at({r, c, d})->setScalarValue(generated);
            }
        }
    }
}