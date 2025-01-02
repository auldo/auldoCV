#pragma once

#include "data/base_tensor.h"

#define CIFAR_10_BATCH_1 "data_batch_1.bin"
#define CIFAR_10_BATCH_2 "data_batch_2.bin"
#define CIFAR_10_BATCH_3 "data_batch_3.bin"
#define CIFAR_10_BATCH_4 "data_batch_4.bin"
#define CIFAR_10_BATCH_5 "data_batch_5.bin"
#define CIFAR_10_BATCH_TEST "test_batch.bin"

/// Reads cifar 10 batches that can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html.
std::pair<BASE_TENSOR_REF(PIXEL), VECTOR_REF(PIXEL)> readCifar10Batch(const std::string& path, const std::string& batchName);