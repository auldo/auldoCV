#include "compute_graph.h"

ComputeNodeValue cg_forward(ComputeNodeRef node) {
    if(cnIsConstantOrVariable(node))
        return cnUnwrap(node);

    if(node->first != NULL)
        cnSetCache(1, node, cg_forward(node->first));

    if(node->second != NULL)
        cnSetCache(2, node, cg_forward(node->second));

    ComputeNodeOperatorType op = cnGetOperator(node);

    if(op == CN_OP_SUM || op == CN_OP_SUM_CONST)
        return cnGetCache(1, node) + cnGetCache(2, node);

    if(op == CN_OP_PRODUCT || op == CN_OP_PRODUCT_CONST)
        return cnGetCache(1, node) * cnGetCache(2, node);

    THROW("bad operator");
}