#ifndef AULDO_CV_COMPUTE_NODE_H
#define AULDO_CV_COMPUTE_NODE_H

#include <stdio.h>
#include <memory.h>
#include <stdbool.h>
#include <malloc/_malloc.h>
#include <_stdlib.h>

#define ComputeNodeRef struct ComputeNode*
#define ConstComputeNodeRef const ComputeNodeRef

#define VarComputeNodeCache struct VariableComputeNodeCache
#define OpComputeNodeCache struct OperatorComputeNodeCache

#define ComputeNodeValue double
#define ComputeNodeOperatorType unsigned int

#define THROW(x) { printf((const char *) stderr, x); exit(0); }

#define CN_OP_SUM 0
#define CN_OP_SUM_CONST 1
#define CN_OP_PRODUCT 2
#define CN_OP_PRODUCT_CONST 3

#define CONST(x) cnCreateConstant(x)
#define VAR(x) cnCreateVariable(x)

#define ADD(x, y) cnCreateOperator(CN_OP_SUM, x, y)
//#define SUM_CONST(x, y) cnCreateOperator(CN_OP_SUM_CONST, x CONST(x))

VarComputeNodeCache {
    void *gradient;
};

OpComputeNodeCache {
    void* first;
    void* second;
    void* gradient;
};

/// A compute node references a numeric value or a numeric value operator / function with light overhead.
/// This enables us to use the node in a graph (= mathematical more complex term), i.e. reasonable combination of values and operators.
/// This again enables us to calculate results (forward run) and differentiate back (backward run) the result with respect to certain values.
struct ComputeNode {

    /// If op == NULL this may be the scalar represented by this node (double*)
    void* first;

    /// NULL if this node is a scalar or constant
    /// For a CN to be interpreted as operation this must be the operator
    void* op;

    /// If op == NULL this may be the constant represented by this node (double*)
    /// If op != NULL this may be the second operand (struct auldo_cv_compute_node*)
    void* second;

    /// NULL for constant.
    /// Scalar cache for scalar containing only the gradient.
    /// Full cache for operator nodes.
    void* cache;

};

// General

/// Frees a compute node and all of its resources.
void cnFree(ComputeNodeRef node);

/// Asserts a pointer is null and throws if not.
void cnAssertNull(const void* ptr);

/// Inits the node cache.
/// If node is scalar, the cache will only hold a gradient.
/// A const node does not have a cache.
/// An operator node has caches for saving results from 2 incoming branches and gradient.
void cnInitCache(ComputeNodeRef node);

/// Sets value in cache
/// 0 for gradient, 1 and 2 for first and second branch cache.
void cnSetCache(int address, ComputeNodeRef, ComputeNodeValue value);

/// Gets value from cache.
/// For address, see set_cache documentation.
ComputeNodeValue cnGetCache(int address, ComputeNodeRef node);

// Scalars

/// Creates a variable compute node.
ComputeNodeRef cnCreateVariable(ComputeNodeValue val);

/// Checks if compute node is a variable compute node.
bool cnIsVariable(ConstComputeNodeRef node);

/// Checks if compute node is a variable compute node and throws if not.
void cnAssertVariable(ConstComputeNodeRef node);

// Constants

/// Creates a constant compute node.
ComputeNodeRef cnCreateConstant(ComputeNodeValue val);

/// Checks if compute node is a constant compute node.
bool cnIsConstant(ConstComputeNodeRef node);

/// Checks if compute node is a constant compute node and throws if not.
void cnAssertConstant(ConstComputeNodeRef node);

/// Throws if compute node is a constant compute node.
void cnAssertNoConstant(ConstComputeNodeRef node);

// Scalars & Contants

/// Returns true iff compute node is a leaf (i.e., scalar or constant node).
bool cnIsConstantOrVariable(ConstComputeNodeRef node);

/// Throws if compute node is an operator node.
void cnAssertConstantOrVariable(ConstComputeNodeRef node);

/// Retrieves value from a scalar or constant operator node.
ComputeNodeValue cnUnwrap(ConstComputeNodeRef node);

/// Modifies value of a scalar or constant operator node.
void cnSet(ComputeNodeRef node, ComputeNodeValue val);

// Operators

/// Validates inputs with respect to the operator.
void cnValidateOpInputs(ComputeNodeOperatorType operator, ComputeNodeRef first, ComputeNodeRef second);

/// Creates the compute node.
ComputeNodeRef cnCreateOperator(ComputeNodeOperatorType operator, ComputeNodeRef first, ComputeNodeRef second);

/// Returns the operator of an operator compute node.
ComputeNodeOperatorType cnGetOperator(ConstComputeNodeRef node);

/// Returns true iff the node is an operator node.
bool cnIsOperator(ConstComputeNodeRef node);

/// Throws if the node is no operator node.
void cnAssertOperator(ConstComputeNodeRef node);

#endif //AULDO_CV_COMPUTE_NODE_H