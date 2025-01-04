#include "loss/binary_cross_entropy_loss.h"

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(PRECISE_NBR label, const PTR<ComputeNode>& prediction) {
    auto truth{COMPUTE_NODE(label)};
    auto negPrediction{COMPUTE_NODE_TIMES_CONST(prediction, -1)};
    auto increasedNegPred{COMPUTE_NODE_PLUS_CONST(negPrediction, 1)};
    auto logIncrNegPred{COMPUTE_NODE_LOG_N(increasedNegPred)};
    auto negTruth{COMPUTE_NODE_TIMES_CONST(truth, -1)};
    auto increasedNegTruth{COMPUTE_NODE_PLUS_CONST(negTruth, 1)};
    auto rightSummand{COMPUTE_NODE_TIMES(increasedNegTruth, logIncrNegPred)};

    auto logPrediction{COMPUTE_NODE_LOG_N(prediction)};
    auto leftSummand{COMPUTE_NODE_TIMES(truth, logPrediction)};

    auto sum {COMPUTE_NODE_PLUS(leftSummand, rightSummand)};
    _output_node = COMPUTE_NODE_TIMES_CONST(sum, -1);
}