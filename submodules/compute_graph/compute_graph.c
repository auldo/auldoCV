#include "compute_graph.h"

CN_TYPE run_compute_graph_forward(CN_PTR node) {
    if(compute_node_is_scalar_or_constant(node))
        return value(node);

    if(node->first != NULL)
        set_cache(1, node, run_compute_graph_forward(node->first));

    if(node->second != NULL)
        set_cache(2, node, run_compute_graph_forward(node->second));

    CN_OP_TYPE op = get_operator(node);

    if(op == CN_OP_SUM || op == CN_OP_SUM_CONST)
        return get_cache(1, node) + get_cache(2, node);

    if(op == CN_OP_PRODUCT || op == CN_OP_PRODUCT_CONST)
        return get_cache(1, node) * get_cache(2, node);

    THROW("bad operator");
}