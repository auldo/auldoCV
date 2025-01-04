#include "loss/mse_loss.h"

MSELoss::MSELoss(PRECISE_NBR label, const PTR<ComputeNode>& node) {
    auto neg{COMPUTE_NODE_TIMES_CONST(node, -1)};
    auto sum{COMPUTE_NODE_PLUS_CONST(neg, label)};
    _output_node = COMPUTE_NODE_POW(sum, 2);
}
