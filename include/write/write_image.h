#ifndef INCLUDE_WRITE_WRITE_IMAGE_H
#define INCLUDE_WRITE_WRITE_IMAGE_H

#include "tensor.h"
#include <opencv2/opencv.hpp>

std::shared_ptr<cv::Mat> write_image(const std::shared_ptr<tensor<int>>& data, int type, const std::string& filename);

#endif