#include "dataset/read_cifar_10.h"

#include <fstream>
#include <iostream>
#include <data/base_tensor.h>

std::pair<BASE_TENSOR_REF(PIXEL), VECTOR_REF(PIXEL)> readCifar10Batch(const std::string& path, const std::string& batchName) {
    std::ifstream source(path + "/" + batchName, std::ios_base::binary);
    const std::shared_ptr tensor{std::make_shared<BaseTensor<PIXEL>>(Vector<INDEX_NBR>({10000, 32, 32, 3}))};
    const auto vector{std::make_shared<Vector<PIXEL>>(10000)};
    for(INDEX_NBR i{0}; i < 10000; ++i) {
        const PIXEL label = source.get();
        vector->at(i) = label;
        for(INDEX_NBR channel{0}; channel < 3; ++channel) {
            for(INDEX_NBR r{0}; r < 32; ++r) {
                for(INDEX_NBR c{0}; c < 32; ++c) {
                    tensor->at({i, r, c, channel}) = source.get();
                }
            }
        }
    }
    return std::make_pair(tensor, vector);
}
