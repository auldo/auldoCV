#pragma once

#include "data/tensor.h"
#include "concept/arithmetic.h"

template <typename T> requires arithmetic<T>
TENSOR_REF(PIXEL) convertPixelsToByte(TENSOR_REF(T));

template <typename T> requires arithmetic<T>
TENSOR_REF(PIXEL) convertPixelsToByte(TENSOR_REF(T) input) {
    if(input->_dimensions.size() != 3)
        throw std::invalid_argument("Input must be a 3d tensor");
    auto dimensions{input->_dimensions};
    auto result{std::make_shared<BaseTensor<PIXEL>>(dimensions)};

    for(INDEX_NBR r{0}; r < result->shapeSize(0); ++r) {
        for(INDEX_NBR c{0}; c < result->shapeSize(1); ++c) {
            for(INDEX_NBR ch{0}; ch < result->shapeSize(2); ++ch) {
                PIXEL p{TO_PIXEL(input->at({r, c, ch}))};
                result->at({r, c, ch}) = p;
            }
        }
    }

    return result;
}