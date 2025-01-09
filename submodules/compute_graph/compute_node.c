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

void init_cache(CN_PTR node) {
    if(node->cache != NULL)
        free(node->cache);
    if(compute_node_is_constant(node))
        return;
    if(compute_node_is_scalar(node)) {
        struct CN_SCALAR_CACHE* ptr = malloc(sizeof(struct CN_SCALAR_CACHE));
        ptr->gradient = NULL;
        node->cache = ptr;
        return;
    }
    struct CN_CACHE* ptr = malloc(sizeof(struct CN_CACHE));
    ptr->first = NULL;
    ptr->second = NULL;
    ptr->gradient = NULL;
    node->cache = ptr;
}

void set_cache(int address, CN_PTR node, CN_TYPE value) {

    if(address != 0 && address != 1 && address != 2)
        THROW("bad cache address");

    if(address == 0) {
        if(node->cache == NULL)
            THROW("expected cache");

        if(compute_node_is_scalar(node)) {
            struct CN_SCALAR_CACHE* cache = node->cache;
            if(cache->gradient == NULL)
                cache->gradient = malloc(sizeof(CN_TYPE));
            *(CN_TYPE*) cache->gradient = value;
            return;
        }
        struct CN_CACHE* cache = node->cache;
        if(cache->gradient == NULL)
            cache->gradient = malloc(sizeof(CN_TYPE));
        *(CN_TYPE*) cache->gradient = value;
        return;
    }

    if(node->cache == NULL || compute_node_is_scalar(node))
        THROW("expected full cache");
    struct CN_CACHE* cache = node->cache;

    if(address == 1) {
        if(cache->first == NULL)
            cache->first = malloc(sizeof(CN_TYPE));
        *(CN_TYPE*) cache->first = value;
        return;
    }

    if(cache->second == NULL)
        cache->second = malloc(sizeof(CN_TYPE));
    *(CN_TYPE*) cache->second = value;
}

CN_TYPE get_cache(int address, CN_PTR node) {

    if(address != 0 && address != 1 && address != 2)
        THROW("bad cache address");

    if(address == 0) {
        if(node->cache == NULL)
            THROW("expected cache");

        if(compute_node_is_scalar(node)) {
            struct CN_SCALAR_CACHE* cache = node->cache;
            if(cache->gradient == NULL)
                THROW("expected gradient");
            return *(CN_TYPE*) cache->gradient;
        }
        struct CN_CACHE* cache = node->cache;
        if(cache->gradient == NULL)
            THROW("expected gradient");
        return *(CN_TYPE*) cache->gradient;
    }

    if(node->cache == NULL || compute_node_is_scalar(node))
        THROW("expected full cache");
    struct CN_CACHE* cache = node->cache;

    if(address == 1) {
        if(cache->first == NULL)
            THROW("expected first cache");
        return *(CN_TYPE*) cache->first;
    }

    if(cache->second == NULL)
        THROW("expected second cache");
    return *(CN_TYPE*) cache->second;
}

// Scalars

CN_PTR scalar(CN_TYPE val) {
    CN_TYPE* ptr = malloc(sizeof(CN_TYPE));
    *ptr = val;
    CN_PTR node = malloc(sizeof(struct auldo_cv_compute_node));
    node->first = ptr;
    node->second = NULL;
    node->op = NULL;
    init_cache(node);
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
    node->first = first;
    node->second = second;
    node->op = opPtr;

    init_cache(node);

    return node;
}

CN_OP_TYPE get_operator(const CN_PTR node) {
    assert_operator(node);
    return *(CN_OP_TYPE*) node->op;
}

CN_BOOL_TYPE compute_node_is_operator(const CN_PTR node) {
    return node->op != NULL;
}

void assert_operator(const CN_PTR node) {
    if(!compute_node_is_operator(node))
        THROW("expected operator node");
}