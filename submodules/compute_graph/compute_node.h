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
void free_compute_node(CN_PTR node);
void assert_null(const void* ptr);
void init_cache(CN_PTR node);
void set_cache(int address, CN_PTR node, CN_TYPE value);
CN_TYPE get_cache(int address, CN_PTR node);

// Scalars
CN_PTR scalar(CN_TYPE val);
CN_BOOL_TYPE compute_node_is_scalar(const CN_PTR node);
void assert_scalar(const CN_PTR node);

// Constants
CN_PTR constant(CN_TYPE val);
CN_BOOL_TYPE compute_node_is_constant(const CN_PTR node);
void assert_constant(const CN_PTR node);
void assert_no_constant(const CN_PTR node);

// Scalars & Contants
CN_BOOL_TYPE compute_node_is_scalar_or_constant(const CN_PTR node);
void assert_scalar_or_constant(const CN_PTR node);
CN_TYPE value(const CN_PTR node);
void set_value(CN_PTR node, CN_TYPE val);

// Operators
void _validate_operator_inputs(CN_OP_TYPE operator, CN_PTR first, CN_PTR second);
CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, CN_PTR second);
CN_OP_TYPE get_operator(const CN_PTR node);
CN_BOOL_TYPE compute_node_is_operator(const CN_PTR node);
void assert_operator(const CN_PTR node);

#endif //AULDO_CV_COMPUTE_NODE_H