#include "vision/base_kernel.h"

BaseKernel::BaseKernel(PIXEL stride, PIXEL size, std::optional<PIXEL> padding): _stride(stride), _padding(padding), _size(size) {}