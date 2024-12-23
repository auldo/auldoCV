#ifndef INCLUDE_COMPUTATIONAL_GRAPH_COMPUTATIONAL_GRAPH_NODE_H
#define INCLUDE_COMPUTATIONAL_GRAPH_COMPUTATIONAL_GRAPH_NODE_H

#include <optional>
#include <vector>
#include <iostream>

enum ComputationalGraphNodeType {
    OP_PLUS, OP_TIMES,
    SCALAR
};

class ComputationalGraphNode {
    ComputationalGraphNodeType type;

    std::vector<std::shared_ptr<ComputationalGraphNode>> incoming;
    std::optional<float> scalar;
    std::vector<float> _gradients;
public:
    std::vector<float> forwardPassCache{};
    explicit ComputationalGraphNode(float scalar);
    ComputationalGraphNode(ComputationalGraphNodeType op, std::vector<std::shared_ptr<ComputationalGraphNode>> incoming);
    float forwardPass();
    void backwardPass(std::optional<float> current);
    [[nodiscard]] float derivative(unsigned long idx) const;
    [[nodiscard]] float gradient() const;
};

#endif //INCLUDE_COMPUTATIONAL_GRAPH_COMPUTATIONAL_GRAPH_NODE_H