#include "gradient/compute_node.h"

ComputeNode::ComputeNode(PRECISE_NBR scalar): _scalar(scalar), _type(SCALAR) {}

ComputeNode::ComputeNode(ComputeNodeType op, ComputeNodeDynamicArgs dynamicArgs): _type(op), _dynamicArgs(std::move(dynamicArgs)) {
    if(op == SCALAR)
        throw std::runtime_error("SCALAR type not allowed as operator.");
}

ComputeNode::ComputeNode(const ComputeNodeType op, ComputeNodeConstArgs constArgs): _type(op), _constArgs(std::move(constArgs)) {}

ComputeNode::ComputeNode(ComputeNodeType op, ComputeNodeDynamicArgs dynamicArgs, ComputeNodeConstArgs constArgs): _type(op), _dynamicArgs(std::move(dynamicArgs)), _constArgs(std::move(constArgs)) {}


PRECISE_NBR ComputeNode::forwardPass() {
    clear();
    for(const auto& node : _dynamicArgs)
        forwardPassCache.push_back(node->forwardPass());

    if(_type == SCALAR)
        return _scalar.value();

    if(_type == OP_PLUS)
        return forwardPassCache.at(0) + forwardPassCache.at(1);

    if(_type == OP_TIMES)
        return forwardPassCache.at(0) * forwardPassCache.at(1);

    if(_type == OP_TIMES_CONST)
        return forwardPassCache.at(0) * _constArgs.at(0);

    if(_type == FN_EXP)
        return exp(forwardPassCache.at(0));

    if(_type == OP_PLUS_CONST)
        return forwardPassCache.at(0) + _constArgs.at(0);

    if(_type == OP_DIV_CONST_DIVIDEND)
        return _constArgs.at(0) / forwardPassCache.at(0);

    if(_type == FN_RELU)
        return forwardPassCache.at(0) > 0 ? forwardPassCache.at(0) : 0;

    if(_type == OP_POW_CONST_EXPONENT)
        return pow(forwardPassCache.at(0), _constArgs.at(0));

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
        return _constArgs.at(0);

    if(_type == FN_EXP)
        return exp(forwardPassCache.at(0));

    if(_type == OP_PLUS_CONST)
        return 1;

    /*
     * Partial derivative d(c / x) / d(x) = (-1 * c) / x^2
     */
    if(_type == OP_DIV_CONST_DIVIDEND)
        return (-1.f * _constArgs.at(0)) / static_cast<float>(pow(forwardPassCache.at(0), 2));

    if(_type == FN_RELU)
        return forwardPassCache.at(0) > 0 ? 1 : 0;

    if(_type == OP_POW_CONST_EXPONENT)
        return _constArgs.at(0) * pow(forwardPassCache.at(0), _constArgs.at(0) - 1);

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
    if(_gradient == std::nullopt)
        _gradient = der;
    else
        _gradient = _gradient.value() + der;
    for(INDEX_NBR i{0}; i < _dynamicArgs.size(); ++i) {
        _dynamicArgs.at(i)->backwardPass(der * derivative(i));
    }
}

PRECISE_NBR ComputeNode::gradient() {
    if(_gradient == std::nullopt)
        throw std::runtime_error("no gradients in computational graph");
    PRECISE_NBR result{0};
    auto gradient{_gradient.value()};
    _gradient = std::nullopt;
    return gradient;
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
    for(auto& elem : _dynamicArgs)
        elem->clear();
    forwardPassCache.clear();
    _gradient = std::nullopt;
}

void ComputeNode::applyGradient(PRECISE_NBR factor) {
    setScalarValue(getScalarValue() - factor * gradient());
}

void ComputeNode::rescaleGradientStorage(INDEX_NBR size) {
    _gradientStorage = Vector<PRECISE_NBR>(size);
}

void ComputeNode::setGradientStorage(INDEX_NBR index) {
    _gradientStorage.at(index) = gradient();
}

void ComputeNode::applyAverageGradient(double factor) {
    PRECISE_NBR sum{0};
    for(const auto& elem : _gradientStorage)
        sum += elem;
    PRECISE_NBR weightedSum{sum / _gradientStorage.size()};
    setScalarValue(getScalarValue() - factor * weightedSum);
}

