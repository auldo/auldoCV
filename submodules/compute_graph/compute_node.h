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

#define CN_OP_PLUS 0

struct CN {
    /// For a CN to be interpreted as scalar, this must be a double*
    /// If _arg2 != NULL this must be the first operand
    void* first;

    /// NULL if _arg1 is a scalar
    /// For a CN to be interpreted as operation this must be the operator
    void* op;

    /// If _arg2 != NULL this may be the second operand
    void* second;
};

CN_PTR create_scalar_compute_node(CN_TYPE val);
CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, CN_PTR second);
void free_compute_node(CN_PTR node);
CN_BOOL_TYPE compute_node_is_scalar(const CN_PTR node);
CN_TYPE scalar(CN_PTR node);
void set_scalar(CN_PTR node, CN_TYPE val);
void assert_scalar(CN_PTR node);

#endif //AULDO_CV_COMPUTE_NODE_H