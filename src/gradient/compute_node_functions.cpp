#include "gradient/compute_node_functions.h"

PRECISE_NBR sigmoid(PRECISE_NBR x) {
    return 1 / (1 + exp(-1*x));
}

PRECISE_NBR sigmoidDerivative(PRECISE_NBR x) {
    PRECISE_NBR tmp{sigmoid(x)};
    return tmp * (1 - tmp);
}