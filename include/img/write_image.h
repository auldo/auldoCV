#pragma once

#include <opencv2/opencv.hpp>
#include "data/tensor.h"

void writeImage(const std::string& path, const TENSOR_REF(PIXEL)& data);