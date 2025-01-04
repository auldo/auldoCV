#pragma once

#include "data/base_tensor.h"
#include "fstream"

#define CIFAR_10_BATCH_1 "data_batch_1.bin"
#define CIFAR_10_BATCH_2 "data_batch_2.bin"
#define CIFAR_10_BATCH_3 "data_batch_3.bin"
#define CIFAR_10_BATCH_4 "data_batch_4.bin"
#define CIFAR_10_BATCH_5 "data_batch_5.bin"
#define CIFAR_10_BATCH_TEST "test_batch.bin"

/// Reads cifar 10 batches that can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html.
template <typename T>
std::pair<TENSOR_REF(T), TENSOR_REF(T)> readCifar10Batch(const std::string& path, const std::string& batchName);

template <typename T>
std::pair<TENSOR_REF(T), TENSOR_REF(T)> readCifar10Batch(const std::string& path, const std::string& batchName) {
    std::ifstream source(path + "/" + batchName, std::ios_base::binary);
    const auto tensor{std::make_shared<BaseTensor<T>>(Vector<INDEX_NBR>({10000, 32, 32, 3}))};
    const auto labels{std::make_shared<BaseTensor<T>>(Vector<INDEX_NBR>({10000, 1}))};
    for(INDEX_NBR i{0}; i < 10000; ++i) {
        const PIXEL label = source.get();
        labels->at({i, 0}) = label;
        for(INDEX_NBR channel{0}; channel < 3; ++channel) {
            for(INDEX_NBR r{0}; r < 32; ++r) {
                for(INDEX_NBR c{0}; c < 32; ++c) {
                    tensor->at({i, r, c, channel}) = static_cast<T>(source.get());
                }
            }
        }
    }
    return std::make_pair(tensor, labels);
}