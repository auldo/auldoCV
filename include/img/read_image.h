#pragma once

#include <opencv2/opencv.hpp>
#include "data/tensor.h"

TENSOR_REF(PIXEL) readImage(const std::string& path);