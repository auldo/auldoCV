#include "compute_node.h"

// General

void cnFree(ComputeNodeRef node) {
    if(node == NULL)
        return;
    free(node->first);
    free(node->second);
    free(node->op);
    if(node->cache != NULL)
        free(node->cache);
    free(node);
}

void cnAssertNull(const void* ptr) {
    if(ptr != NULL)
        THROW("expected null");
}

void cnInitCache(ComputeNodeRef node) {
    if(node->cache != NULL)
        free(node->cache);
    if(cnIsConstant(node))
        return;
    if(cnIsVariable(node)) {
        VarComputeNodeCache* ptr = malloc(sizeof(VarComputeNodeCache));
        ptr->gradient = NULL;
        node->cache = ptr;
        return;
    }
    OpComputeNodeCache* ptr = malloc(sizeof(OpComputeNodeCache));
    ptr->first = NULL;
    ptr->second = NULL;
    ptr->gradient = NULL;
    node->cache = ptr;
}

void cnSetCache(int address, ComputeNodeRef node, ComputeNodeValue value) {

    if(address != 0 && address != 1 && address != 2)
        THROW("bad cache address");

    if(address == 0) {
        if(node->cache == NULL)
            THROW("expected cache");

        if(cnIsVariable(node)) {
            VarComputeNodeCache* cache = node->cache;
            if(cache->gradient == NULL)
                cache->gradient = malloc(sizeof(ComputeNodeValue));
            *(ComputeNodeValue*) cache->gradient = value;
            return;
        }
        OpComputeNodeCache* cache = node->cache;
        if(cache->gradient == NULL)
            cache->gradient = malloc(sizeof(ComputeNodeValue));
        *(ComputeNodeValue*) cache->gradient = value;
        return;
    }

    if(node->cache == NULL || cnIsVariable(node))
        THROW("expected full cache");
     OpComputeNodeCache* cache = node->cache;

    if(address == 1) {
        if(cache->first == NULL)
            cache->first = malloc(sizeof(ComputeNodeValue));
        *(ComputeNodeValue*) cache->first = value;
        return;
    }

    if(cache->second == NULL)
        cache->second = malloc(sizeof(ComputeNodeValue));
    *(ComputeNodeValue*) cache->second = value;
}

ComputeNodeValue cnGetCache(int address, ComputeNodeRef node) {

    if(address != 0 && address != 1 && address != 2)
        THROW("bad cache address");

    if(address == 0) {
        if(node->cache == NULL)
            THROW("expected cache");

        if(cnIsVariable(node)) {
            VarComputeNodeCache* cache = node->cache;
            if(cache->gradient == NULL)
                THROW("expected gradient");
            return *(ComputeNodeValue*) cache->gradient;
        }
        OpComputeNodeCache* cache = node->cache;
        if(cache->gradient == NULL)
            THROW("expected gradient");
        return *(ComputeNodeValue*) cache->gradient;
    }

    if(node->cache == NULL || cnIsVariable(node))
        THROW("expected full cache");
    OpComputeNodeCache* cache = node->cache;

    if(address == 1) {
        if(cache->first == NULL)
            THROW("expected first cache");
        return *(ComputeNodeValue*) cache->first;
    }

    if(cache->second == NULL)
        THROW("expected second cache");
    return *(ComputeNodeValue*) cache->second;
}

// Scalars

ComputeNodeRef cnCreateVariable(ComputeNodeValue val) {
    ComputeNodeValue* ptr = malloc(sizeof(ComputeNodeValue));
    *ptr = val;
    ComputeNodeRef node = malloc(sizeof(struct ComputeNode));
    node->first = ptr;
    node->second = NULL;
    node->op = NULL;
    cnInitCache(node);
    return node;
}

bool cnIsVariable(ConstComputeNodeRef node) {
    return node->op == NULL && node->first != NULL;
}

void cnAssertVariable(ConstComputeNodeRef node) {
    if(!cnIsVariable(node))
        THROW("expected scalar");
}

// Constants

ComputeNodeRef cnCreateConstant(ComputeNodeValue val) {
    ComputeNodeValue* ptr = malloc(sizeof(ComputeNodeValue));
    *ptr = val;
    ComputeNodeRef node = malloc(sizeof(struct ComputeNode));
    node->first = NULL;
    node->second = ptr;
    node->op = NULL;
    return node;
}

bool cnIsConstant(ConstComputeNodeRef node) {
    return node->op == NULL && node->second != NULL;;
}

void cnAssertConstant(ConstComputeNodeRef node) {
    if(!cnIsConstant(node))
        THROW("expected constant");
}

void cnAssertNoConstant(ConstComputeNodeRef node) {
    if(cnIsConstant(node))
        THROW("expected no constant");
}

// Scalars & Constants

bool cnIsConstantOrVariable(ConstComputeNodeRef node) {
    return cnIsVariable(node) || cnIsConstant(node);
}

void cnAssertConstantOrVariable(ConstComputeNodeRef node) {
    if(!cnIsConstantOrVariable(node))
        THROW("expected scalar or constant");
}

ComputeNodeValue cnUnwrap(ConstComputeNodeRef node) {
    cnAssertConstantOrVariable(node);
    return *(cnIsVariable(node) ? (ComputeNodeValue*) node->first : (ComputeNodeValue*) node->second);
}

void cnSet(ComputeNodeRef node, ComputeNodeValue val) {
    cnAssertConstantOrVariable(node);
    *(cnIsVariable(node) ? (ComputeNodeValue*) node->first : (ComputeNodeValue*) node->second) = val;
}

// Operators

void cnValidateOpInputs(ComputeNodeOperatorType operator, ComputeNodeRef first, ComputeNodeRef second) {
    if(operator == CN_OP_SUM || operator == CN_OP_PRODUCT) {
        cnAssertNoConstant(first);
        cnAssertNoConstant(second);
        return;
    }
    if(operator == CN_OP_SUM_CONST || operator == CN_OP_PRODUCT_CONST) {
        cnAssertNoConstant(first);
        cnAssertConstant(second);
        return;
    }
    THROW("validation failed")
}

ComputeNodeRef cnCreateOperator(ComputeNodeOperatorType operator, ComputeNodeRef first, ComputeNodeRef second) {
    cnValidateOpInputs(operator, first, second);

    ComputeNodeOperatorType* opPtr = malloc(sizeof(ComputeNodeOperatorType));
    *opPtr = operator;

    ComputeNodeRef node = malloc(sizeof(struct ComputeNode));
    node->first = first;
    node->second = second;
    node->op = opPtr;

    cnInitCache(node);

    return node;
}

ComputeNodeOperatorType cnGetOperator(const ComputeNodeRef node) {
    cnAssertOperator(node);
    return *(ComputeNodeOperatorType*) node->op;
}

bool cnIsOperator(const ComputeNodeRef node) {
    return node->op != NULL;
}

void cnAssertOperator(const ComputeNodeRef node) {
    if(!cnIsOperator(node))
        THROW("expected operator node");
}