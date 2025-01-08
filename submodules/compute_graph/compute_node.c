#include "compute_node.h"

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
    return node->op == NULL;
}

void free_compute_node(CN_PTR node) {
    free(node->first);
    free(node->second);
    free(node->op);
    free(node);
}

CN_TYPE scalar(CN_PTR node) {
    if(!compute_node_is_scalar(node))
        THROW("expected scalar");
    return *((CN_TYPE*) node->first);
}

CN_PTR create_operator_compute_node(CN_OP_TYPE operator, CN_PTR first, CN_PTR second) {
    CN_TYPE* op = malloc(sizeof(CN_OP_TYPE));
    *op = operator;
    CN_PTR node = malloc(sizeof(struct auldo_cv_compute_node));
    node->first = first;
    node->second = second;
    node->op = op;
    return node;
}