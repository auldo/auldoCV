#include "compute_node.h"

// General

void free_compute_node(CN_PTR node) {
    free(node->first);
    free(node->second);
    free(node->op);
    free(node);
}

// Scalars

CN_PTR create_scalar_compute_node(CN_TYPE val) {
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

CN_PTR create_constant_compute_node(CN_TYPE val) {
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

CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, void* second) {
    THROW("validation failed")
}