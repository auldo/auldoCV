#pragma once
#include <optional>
#include <vector>
#include <iostream>
#include <data/vector.h>

#include "constants.h"
#include "gradient/compute_node_type.h"
#include <gradient/compute_node_functions.h>

/// A node in a computational graph.
/// Primarily used to apply the chain rule to complex functions.
class ComputeNode {

    using ComputeNodeDynamicArgs = Vector<std::shared_ptr<ComputeNode>>;
    using ComputeNodeConstArgs = Vector<PRECISE_NBR>;

    /// Describes the type, which may be a function, scalar or operator involving one or multiple constants and variables.
    ComputeNodeType _type;

    /// The list of constants used by the node.
    /// E.g., the OP_PLUS_CONST type requires one variable and one const summand.
    Vector<PRECISE_NBR> _constArgs;

    /// The vector of refs to incoming computational graph nodes.
    /// Required for recursive forward pass and backward pass.
    Vector<std::shared_ptr<ComputeNode>> _dynamicArgs;

    /// Holds a value if type is scalar.
    /// Primarily used for variable inputs to the graph.
    /// Shouldn't be used for const values.
    std::optional<PRECISE_NBR> _scalar;

    /// Stores the gradient from the backward pass.
    /// Those are added up.
    std::optional<PRECISE_NBR> _gradient;

    Vector<PRECISE_NBR> _gradientStorage;

public:

    explicit ComputeNode(ComputeNodeType type);

    /// Creates a computational graph node of SCALAR type.
    explicit ComputeNode(PRECISE_NBR scalar);

    /// Creates a computational node of any type with children graph nodes.
    ComputeNode(ComputeNodeType op, ComputeNodeDynamicArgs dynamicArgs);

    /// Creates a computational node of any type with const args.
    ComputeNode(ComputeNodeType op, ComputeNodeConstArgs constArgs);

    /// Creates a computational node of any type with children graph nodes and const args.
    ComputeNode(ComputeNodeType op, ComputeNodeDynamicArgs dynamicArgs, ComputeNodeConstArgs constArgs);

    /// The cache of the forward pass.
    std::vector<PRECISE_NBR> forwardPassCache{};

    /// Recursively navigates to the graph's leaves and starts the forward pass.
    /// Adds incoming graph nodes' forward pass results to the forward pass cache.
    /// Returns the own forward pass result depending on the children nodes' results.
    PRECISE_NBR forwardPass();

    /// Calls overload backwardPass function with null_optional.
    void backwardPass();

    /// Runs the backward pass recursively.
    /// Passes along the current node's derivative multiplied by incoming backward pass (OR 1 for output node) to the node's children.
    /// Pushes current into the gradient cache.
    void backwardPass(std::optional<PRECISE_NBR> current);

    /// Calculates the derivative of the current node with respect to a variable parameter index.
    /// E.g., for the term a + b, with a and b being variable (thus type OP_PLUS) derivative(0) returns partial derivative d(a+b)/d(a).
    USE_RETURN PRECISE_NBR derivative(INDEX_NBR idx) const;

    /// Sums up the gradients in the gradient cache.
    /// Attention. This clears the gradients you can only access them once.
    /// This mechanism ensures gradients aren't summed up for multiple backward pass iterations.
    /// Store gradients in the higher managing instance, i.e. neuron or layer.
    USE_RETURN PRECISE_NBR gradient();

    /// Throws if node is not of type SCALAR.
    void assertScalar() const;

    /// Asserts scalar and returns the scalar value.
    /// Same effect as calling forwardPass on a scalar typed node but better terminology and scalar assertion.
    USE_RETURN PRECISE_NBR getScalarValue() const;

    /// Updates the scalar value.
    /// This is required because we want a NN to be able to change its weights and biases without rebuilding the whole computational graph.
    void setScalarValue(PRECISE_NBR scalar);

    void clear();

    void applyGradient(PRECISE_NBR factor);

    void rescaleGradientStorage(INDEX_NBR size);
    void setGradientStorage(INDEX_NBR index);
    void applyAverageGradient(PRECISE_NBR factor);
};