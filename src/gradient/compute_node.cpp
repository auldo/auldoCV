#include "gradient/compute_node.h"

ComputeNode::ComputeNode(PRECISE_NBR scalar): _scalar(scalar), _type(SCALAR) {}

ComputeNode::ComputeNode(ComputeNodeType op, std::vector<std::shared_ptr<ComputeNode>> incoming): _type(op), _incoming(std::move(incoming)) {
    if(op == SCALAR)
        throw std::runtime_error("SCALAR type not allowed as operator.");
}

ComputeNode::ComputeNode(const ComputeNodeType op, std::vector<PRECISE_NBR> const_args): _type(op), _const_args(std::move(const_args)) {}

ComputeNode::ComputeNode(ComputeNodeType op, std::vector<std::shared_ptr<ComputeNode> > incoming, std::vector<PRECISE_NBR> const_args): _type(op), _incoming(std::move(incoming)), _const_args(std::move(const_args)) {}


PRECISE_NBR ComputeNode::forwardPass() {
    for(const auto& node : _incoming)
        forwardPassCache.push_back(node->forwardPass());

    if(_type == SCALAR)
        return _scalar.value();

    if(_type == OP_PLUS)
        return forwardPassCache.at(0) + forwardPassCache.at(1);

    if(_type == OP_TIMES)
        return forwardPassCache.at(0) * forwardPassCache.at(1);

    if(_type == OP_TIMES_CONST)
        return forwardPassCache.at(0) * _const_args.at(0);

    if(_type == OP_EXP)
        return exp(forwardPassCache.at(0));

    if(_type == OP_PLUS_CONST)
        return forwardPassCache.at(0) + _const_args.at(0);

    if(_type == OP_DIV_CONST_DIVIDEND)
        return _const_args.at(0) / forwardPassCache.at(0);

    if(_type == FN_RELU)
        return forwardPassCache.at(0) > 0 ? forwardPassCache.at(0) : 0;

    if(_type == OP_POW_CONST_EXPONENT)
        return pow(forwardPassCache.at(0), _const_args.at(0));

    if(_type == OP_DIV)
        return forwardPassCache.at(0) / forwardPassCache.at(1);

    if(_type == FN_LOG_N)
        return std::log(forwardPassCache.at(0));

    if(_type == FN_SIGMOID)
        return sigmoid(forwardPassCache.at(0));

    throw std::runtime_error("bad type");
}

PRECISE_NBR ComputeNode::derivative(INDEX_NBR idx) const {
    if(_type == SCALAR)
        return 0;

    if(_type == OP_TIMES)
        return forwardPassCache.at(idx == 0 ? 1 : 0);

    if(_type == OP_PLUS)
        return 1;

    if(_type == OP_TIMES_CONST)
        return _const_args.at(0);

    if(_type == OP_EXP)
        return exp(forwardPassCache.at(0));

    if(_type == OP_PLUS_CONST)
        return 1;

    /*
     * Partial derivative d(c / x) / d(x) = (-1 * c) / x^2
     */
    if(_type == OP_DIV_CONST_DIVIDEND)
        return (-1.f * _const_args.at(0)) / static_cast<float>(pow(forwardPassCache.at(0), 2));

    if(_type == FN_RELU)
        return forwardPassCache.at(0) > 0 ? 1 : 0;

    if(_type == OP_POW_CONST_EXPONENT)
        return _const_args.at(0) * pow(forwardPassCache.at(0), _const_args.at(0) - 1);

    /*
     * Partial derivative of d(a / b) / d(a) = 1/b
     * Partial derivative of d(a / b) / d(b) = (-1 * a) / b^2
     */
    if(_type == OP_DIV)
        return idx == 0 ? (1 / forwardPassCache.at(1)) : (-1 * forwardPassCache.at(0)) / pow(forwardPassCache.at(1), 2);

    if(_type == FN_LOG_N)
        return 1 / forwardPassCache.at(0);

    if(_type == FN_SIGMOID)
        return sigmoidDerivative(forwardPassCache.at(0));

    throw std::runtime_error("bad type");
}

void ComputeNode::backwardPass() {
    backwardPass(std::optional<float>{});
}

void ComputeNode::backwardPass(std::optional<PRECISE_NBR> current) {
    PRECISE_NBR der;
    if(current == std::nullopt) {
        der = 1;
    } else {
        der = current.value();
    }
    _gradients.push_back(der);
    for(unsigned long i{0}; i < _incoming.size(); ++i) {
        _incoming.at(i)->backwardPass(der * derivative(i));
    }
}

PRECISE_NBR ComputeNode::gradient() {
    if(_gradients.empty())
        throw std::runtime_error("no gradients in computational graph");
    PRECISE_NBR result{0};
    for(auto i{0}; i < _gradients.size(); ++i) {
        result += _gradients[i];
    }
    _gradients.clear();
    return result;
}

void ComputeNode::assertScalar() const {
    if(_type != SCALAR || !_scalar.has_value())
        throw std::runtime_error("no scalar");
}

double ComputeNode::getScalarValue() const {
    return this->_scalar.value();
}

void ComputeNode::setScalarValue(double scalar) {
    this->_scalar = scalar;
}

void ComputeNode::clear() {
    forwardPassCache.clear();
    _gradients.clear();
    for(auto& elem : _incoming)
        elem->clear();
}