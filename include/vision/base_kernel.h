#pragma once

#include <data/base_tensor.h>
#include <img/convert_pixels_precise.h>

#include "concept/arithmetic.h"
#include "data/tensor.h"

class BaseKernel {
protected:
    PIXEL _stride;
    std::optional<PIXEL> _padding;
    PIXEL _size;
public:
    virtual ~BaseKernel() = default;

    BaseKernel(PIXEL stride, PIXEL size, std::optional<PIXEL> padding = std::nullopt);

    template <typename T> requires arithmetic<T>
    std::shared_ptr<Tensor<PRECISE_NBR>> applyPadding(std::shared_ptr<Tensor<T>>);

    USE_RETURN INDEX_NBR calculateOutputDimension(INDEX_NBR input) const;

    virtual std::shared_ptr<Tensor<PRECISE_NBR>> apply(TENSOR_REF(PRECISE_NBR)&);
};

template<typename T> requires arithmetic<T>
std::shared_ptr<Tensor<PRECISE_NBR>> BaseKernel::applyPadding(std::shared_ptr<Tensor<T>> input) {
    if(!_padding.has_value())
        return convertPixelsToPrecise(input);
    PIXEL padding{_padding.value()};
    INDEX_NBR newWidth{input->shapeSize(1) + padding * 2};
    INDEX_NBR newHeight{input->shapeSize(0) + padding * 2};

    auto img{std::make_shared<BaseTensor<PRECISE_NBR>>(Vector({newHeight, newWidth, input->shapeSize(2)}))};

    for(INDEX_NBR channel{0}; channel < input->shapeSize(2); ++channel) {
        for(INDEX_NBR row{0}; row < newHeight; ++row) {
            for(INDEX_NBR col{0}; col < newWidth; ++col) {
                if(row < padding || col < padding || newHeight - row - 1 < padding || newWidth - col - 1 < padding) {
                    img->at({row, col, channel}) = 0;
                } else {
                    img->at({row, col, channel}) = input->at({row - padding, col - padding, channel});
                }
            }
        }
    }
    return img;
}