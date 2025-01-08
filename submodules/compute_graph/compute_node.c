#include "compute_node.h"

// General

void free_compute_node(CN_PTR node) {
    free(node->first);
    free(node->second);
    free(node->op);
    free(node);
}

void assert_null(const void* ptr) {
    if(ptr != NULL)
        THROW("expected null");
}

// Scalars

CN_PTR scalar(CN_TYPE val) {
    CN_TYPE* ptr = malloc(sizeof(CN_TYPE));
    *ptr = val;
    CN_PTR node = malloc(sizeof(struct auldo_cv_compute_node));
    node->first = ptr;
    node->second = NULL;
    node->op = NULL;
    return node;
}

CN_BOOL_TYPE compute_node_is_scalar(const CN_PTR node) {
    return node->op == NULL && node->first != NULL;
}

void assert_scalar(const CN_PTR node) {
    if(!compute_node_is_scalar(node))
        THROW("expected scalar");
}

// Constants

CN_PTR constant(CN_TYPE val) {
    CN_TYPE* ptr = malloc(sizeof(CN_TYPE));
    *ptr = val;
    CN_PTR node = malloc(sizeof(struct auldo_cv_compute_node));
    node->first = NULL;
    node->second = ptr;
    node->op = NULL;
    return node;
}

CN_BOOL_TYPE compute_node_is_constant(const CN_PTR node) {
    return node->op == NULL && node->second != NULL;;
}

void assert_constant(const CN_PTR node) {
    if(!compute_node_is_constant(node))
        THROW("expected constant");
}

void assert_no_constant(const CN_PTR node) {
    if(compute_node_is_constant(node))
        THROW("expected no constant");
}

// Scalars & Constants

CN_BOOL_TYPE compute_node_is_scalar_or_constant(const CN_PTR node) {
    return compute_node_is_scalar(node) || compute_node_is_constant(node);
}

void assert_scalar_or_constant(const CN_PTR node) {
    if(!compute_node_is_scalar_or_constant(node))
        THROW("expected scalar or constant");
}

CN_TYPE value(const CN_PTR node) {
    assert_scalar_or_constant(node);
    return *(compute_node_is_scalar(node) ? (CN_TYPE*) node->first : (CN_TYPE*) node->second);
}

void set_value(CN_PTR node, CN_TYPE val) {
    assert_scalar_or_constant(node);
    *(compute_node_is_scalar(node) ? (CN_TYPE*) node->first : (CN_TYPE*) node->second) = val;
}

// Operators

void _validate_operator_inputs(CN_OP_TYPE operator, CN_PTR first, CN_PTR second) {
    if(operator == CN_OP_SUM || operator == CN_OP_PRODUCT) {
        assert_no_constant(first);
        assert_no_constant(second);
        return;
    }
    if(operator == CN_OP_SUM_CONST || operator == CN_OP_PRODUCT_CONST) {
        assert_no_constant(first);
        assert_constant(second);
        return;
    }
    THROW("validation failed")
}

CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, CN_PTR second) {
    _validate_operator_inputs(operator, first, second);

    CN_OP_TYPE* opPtr = malloc(sizeof(CN_OP_TYPE));
    *opPtr = operator;

    CN_PTR node = malloc(sizeof(struct auldo_cv_compute_node));
    node->first = NULL;
    node->second = NULL;
    node->op = opPtr;

    node->first = first;
    node->second = second;

    return node;
}