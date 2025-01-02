#pragma once

#include "vision/base_kernel.h"

class SimpleKernel: public BaseKernel {
public:
    TENSOR_REF(PRECISE_NBR) _filter;
    SimpleKernel(const Vector<PRECISE_NBR>& filter, PIXEL stride, PIXEL size, std::optional<PIXEL> padding);
    static std::shared_ptr<SimpleKernel> gaussianBlur();
    static std::shared_ptr<SimpleKernel> boxMean();
    static std::shared_ptr<SimpleKernel> edgeDetection();
    static std::shared_ptr<SimpleKernel> sharpen();
    std::shared_ptr<Tensor<double>> apply(TENSOR_REF(PRECISE_NBR)&) override;
};