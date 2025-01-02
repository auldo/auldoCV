#pragma once

#include "concept/arithmetic.h"

#include "data/vector.h"
#include "data/tensor.h"
#include "data/base_tensor.h"
#include "data/nested_tensor.h"
#include "data/linearized.h"

#include "dataset/read_cifar_10.h"

#include "gradient/compute_node.h"
#include "gradient/compute_node_type.h"
#include "gradient/compute_node_functions.h"

#include "img/read_image.h"
#include "img/write_image.h"
#include "img/convert_pixels_byte.h"
#include "img/convert_pixels_precise.h"

#include "vision/base_kernel.h"
#include "vision/simple_kernel.h"

#include "layer/fc_layer.h"
#include "layer/layer.h"

#include "loss/loss.h"
#include "loss/mse_loss.h"

#include "neuron/fc_neuron.h"
#include "neuron/neuron.h"
#include "neuron/activation.h"

#include "constants.h"