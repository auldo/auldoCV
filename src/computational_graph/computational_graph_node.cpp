#include "computational_graph/computational_graph_node.h"

ComputationalGraphNode::ComputationalGraphNode(float scalar): scalar(scalar), type(SCALAR) {}

ComputationalGraphNode::ComputationalGraphNode(ComputationalGraphNodeType op, std::vector<std::shared_ptr<ComputationalGraphNode>> incoming): type(op), incoming(std::move(incoming)) {
    if(op == SCALAR)
        throw std::runtime_error("SCALAR type not allowed as operator.");
}

float ComputationalGraphNode::forwardPass() {
    for(auto& node : incoming)
        forwardPassCache.push_back(node->forwardPass());
    if(type == SCALAR)
        return scalar.value();
    if(type == OP_PLUS)
        return forwardPassCache.at(0) + forwardPassCache.at(1);
    if(type == OP_TIMES)
        return forwardPassCache.at(0) * forwardPassCache.at(1);
    throw std::runtime_error("bad type");
}

float ComputationalGraphNode::derivative(unsigned long idx) const {
    if(type == SCALAR)
        return 0;
    if(type == OP_TIMES)
        return forwardPassCache.at(idx == 0 ? 1 : 0);
    if(type == OP_PLUS)
        return 1;
    return 0;
}

void ComputationalGraphNode::backwardPass(std::optional<float> current) {
    float der;
    if(current == std::nullopt) {
        der = 1;
    } else {
        der = current.value();
    }
    _gradients.push_back(der);
    for(unsigned long i{0}; i < incoming.size(); ++i) {
        incoming.at(i)->backwardPass(der * derivative(i));
    }
}

float ComputationalGraphNode::gradient() const {
    float result{0};
    for(auto i{0}; i < _gradients.size(); ++i) {
        result += _gradients[i];
    }
    return result;
}