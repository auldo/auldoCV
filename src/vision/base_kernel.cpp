#include "vision/base_kernel.h"

BaseKernel::BaseKernel(PIXEL stride, PIXEL size, std::optional<PIXEL> padding): _stride(stride), _padding(padding), _size(size) {}

INDEX_NBR BaseKernel::calculateOutputDimension(INDEX_NBR input) const {
    const INDEX_NBR nominator{input - _size};
    const INDEX_NBR result{TO_INDEX_NBR(floor(TO_PRECISE_NBR(nominator) / TO_PRECISE_NBR(_stride)))};
    return result + 1;
}

