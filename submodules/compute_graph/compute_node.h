#ifndef AULDO_CV_COMPUTE_NODE_H
#define AULDO_CV_COMPUTE_NODE_H

#include <stdio.h>
#include <memory.h>
#include <malloc/_malloc.h>
#include <_stdlib.h>

#define CN auldo_cv_compute_node
#define CN_SCALAR_CACHE auldo_cv_compute_node_scalar_cache
#define CN_CACHE auldo_cv_compute_node_cache

#define CN_PTR struct auldo_cv_compute_node*

#define CN_TYPE double
#define CN_OP_TYPE unsigned int
#define CN_BOOL_TYPE unsigned int

#define THROW(x) { printf((const char *) stderr, x); exit(0); }

#define CN_OP_SUM 0
#define CN_OP_SUM_CONST 1
#define CN_OP_PRODUCT 2
#define CN_OP_PRODUCT_CONST 3

#define CONST(x) constant(x)
#define VAR(x) scalar(x)

#define SUM(x, y) create_operator_compute_node(CN_OP_SUM, x, y)
#define SUM_CONST(x, y) create_operator_compute_node(CN_OP_SUM_CONST, x CONST(x))

struct CN_SCALAR_CACHE {
    void *gradient;
};

struct CN_CACHE {
    void* first;
    void* second;
    void* gradient;
};

/// A compute node references a numeric value or a numeric value operator / function with light overhead.
/// This enables us to use the node in a graph (= mathematical more complex term), i.e. reasonable combination of values and operators.
/// This again enables us to calculate results (forward run) and differentiate back (backward run) the result with respect to certain values.
struct CN {

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
void free_compute_node(CN_PTR node);

/// Asserts a pointer is null and throws if not.
void assert_null(const void* ptr);

/// Inits the node cache.
/// If node is scalar, the cache will only hold a gradient.
/// A const node does not have a cache.
/// An operator node has caches for saving results from 2 incoming branches and gradient.
void init_cache(CN_PTR node);

/// Sets value in cache
/// 0 for gradient, 1 and 2 for first and second branch cache.
void set_cache(int address, CN_PTR node, CN_TYPE value);

/// Gets value from cache.
/// For address, see set_cache documentation.
CN_TYPE get_cache(int address, CN_PTR node);

// Scalars

/// Creates a variable compute node.
CN_PTR scalar(CN_TYPE val);

/// Checks if compute node is a variable compute node.
CN_BOOL_TYPE compute_node_is_scalar(const CN_PTR node);

/// Checks if compute node is a variable compute node and throws if not.
void assert_scalar(const CN_PTR node);

// Constants

/// Creates a constant compute node.
CN_PTR constant(CN_TYPE val);

/// Checks if compute node is a constant compute node.
CN_BOOL_TYPE compute_node_is_constant(const CN_PTR node);

/// Checks if compute node is a constant compute node and throws if not.
void assert_constant(const CN_PTR node);

/// Throws if compute node is a constant compute node.
void assert_no_constant(const CN_PTR node);

// Scalars & Contants

/// Returns true iff compute node is a leaf (i.e., scalar or constant node).
CN_BOOL_TYPE compute_node_is_scalar_or_constant(const CN_PTR node);

/// Throws if compute node is an operator node.
void assert_scalar_or_constant(const CN_PTR node);

/// Retrieves value from a scalar or constant operator node.
CN_TYPE value(const CN_PTR node);

/// Modifies value of a scalar or constant operator node.
void set_value(CN_PTR node, CN_TYPE val);

// Operators

/// Validates inputs with respect to the operator.
void _validate_operator_inputs(CN_OP_TYPE operator, CN_PTR first, CN_PTR second);

/// Creates the compute node.
CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, CN_PTR second);

/// Returns the operator of an operator compute node.
CN_OP_TYPE get_operator(const CN_PTR node);

/// Returns true iff the node is an operator node.
CN_BOOL_TYPE compute_node_is_operator(const CN_PTR node);

/// Throws if the node is no operator node.
void assert_operator(const CN_PTR node);

#endif //AULDO_CV_COMPUTE_NODE_H