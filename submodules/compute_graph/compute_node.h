#ifndef AULDO_CV_COMPUTE_NODE_H
#define AULDO_CV_COMPUTE_NODE_H

#include <stdio.h>
#include <memory.h>
#include <malloc/_malloc.h>
#include <_stdlib.h>

#define CN auldo_cv_compute_node
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

#define PLUS(x, y) create_operator_compute_node(CN_OP_PLUS, x, y)
#define PLUS_CONST(x, y) create_operator_compute_node(CN_OP_PLUS_CONST, x CONST(x))



struct CN {

    /// If op == NULL this may be the scalar represented by this node (double*)
    void* first;

    /// NULL if this node is a scalar or constant
    /// For a CN to be interpreted as operation this must be the operator
    void* op;

    /// If op == NULL this may be the constant represented by this node (double*)
    /// If op != NULL this may be the second operand (struct auldo_cv_compute_node*)
    void* second;
};

// General
void free_compute_node(CN_PTR node);
void assert_null(const void* ptr);

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

#endif //AULDO_CV_COMPUTE_NODE_H