#ifndef INCLUDE_READ_READ_IMAGE_H
#define INCLUDE_READ_READ_IMAGE_H

#include <string>
#include "tensor.h"
#include <opencv2/opencv.hpp>

std::shared_ptr<tensor<int>> read_image(const std::string& image_path);

#endif //INCLUDE_READ_READ_IMAGE_H