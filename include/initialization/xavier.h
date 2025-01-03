#pragma once
#include <random>
#include <data/tensor.h>

#include "data/vector.h"
#include "gradient/compute_node.h"

void xavier(Vector<std::shared_ptr<ComputeNode>>& data, INDEX_NBR n);
void xavier(std::shared_ptr<ComputeNode>& node, INDEX_NBR n);
void xavier(PIXEL& size, PIXEL& depth, std::shared_ptr<Tensor<std::shared_ptr<ComputeNode>>>& data, INDEX_NBR n);