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

void ComputeNode::applyAverageGradient(PRECISE_NBR factor) {
    PRECISE_NBR sum{0};
    for(const auto& elem : _gradientStorage)
        sum += elem;
    PRECISE_NBR weightedSum{sum / _gradientStorage.size()};
    setScalarValue(getScalarValue() - factor * weightedSum);
}

void ComputeNode::applyAverageCloneGradient(PRECISE_NBR factor) {
    PRECISE_NBR sum{gradient()};
    for(const auto& elem : _clones)
        sum += elem->gradient();
    PRECISE_NBR weightedSum{sum / (_clones.size() + 1)};
    setScalarValue(getScalarValue() - factor * weightedSum);
}

ComputeNode::ComputeNode() : _type(OP_PLUS), _gradient(std::nullopt), _scalar(std::nullopt) {}

void ComputeNode::setCloneDepth(INDEX_NBR depth) {
    _clones = Vector<PTR<ComputeNode>>(depth);
    for(auto& elem : _dynamicArgs)
        elem->setCloneDepth(depth);
}


PTR<ComputeNode> ComputeNode::clone(INDEX_NBR depth) {
    auto cloned{std::make_shared<ComputeNode>()};
    cloned->_type = _type;

    cloned->_constArgs = Vector<PRECISE_NBR>(_constArgs.size());
    for(auto i{0}; i < _constArgs.size(); ++i)
        cloned->_constArgs.at(i) = _constArgs.at(i);

    if(_scalar.has_value())
        cloned->_scalar = _scalar.value();

    if(_gradient.has_value())
        cloned->_gradient = _gradient.value();

    cloned->_dynamicArgs = Vector<std::shared_ptr<ComputeNode>>(_dynamicArgs.size());
    for(auto i{0}; i < _dynamicArgs.size(); ++i)
        cloned->_dynamicArgs.at(i) = _dynamicArgs.at(i)->clone(depth);

    _clones.at(depth) = cloned;
    return cloned;
}

void ComputeNode::cloneNetwork(const Vector<PTR<ComputeNode>>& layer, unsigned int depth) {
    auto outputCount{layer.size()};

    auto connector{COMPUTE_NODE(0)};
    for(INDEX_NBR i{0}; i < outputCount; ++i)
        connector = COMPUTE_NODE_PLUS(connector, layer.at(i));

    connector->setCloneDepth(depth);
    for(auto d{0}; d < depth; ++d)
        connector->clone(d);
}

std::shared_ptr<ComputeNode> ComputeNode::operator[](INDEX_NBR index) {
    if(index == 0)
        return shared_from_this();
    if(index > _clones.size())
        throw std::runtime_error("out of bounds");
    return _clones.at(index - 1);
}
