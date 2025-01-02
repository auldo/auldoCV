#pragma once
#include <random>

#include "data/vector.h"
#include "gradient/compute_node.h"

void xavier(Vector<std::shared_ptr<ComputeNode>>& data, INDEX_NBR n);
void xavier(std::shared_ptr<ComputeNode>& node, INDEX_NBR n);