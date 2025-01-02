#pragma once

#include <concept/arithmetic.h>
#include <opencv2/opencv.hpp>
#include "data/base_tensor.h"

template <typename T = PIXEL> requires arithmetic<T>
TENSOR_REF(T) readImage(const std::string& path);

template <typename T> requires arithmetic<T>
TENSOR_REF(T) readImage(const std::string& path) {
    auto img = cv::imread(path, cv::IMREAD_COLOR);
    INDEX_NBR width{TO_INDEX_NBR(img.cols)};
    INDEX_NBR height{TO_INDEX_NBR(img.rows)};
    INDEX_NBR channels{TO_INDEX_NBR(img.channels())};

    //img.convertTo(img, CV_MAKETYPE(CV_8U, channels));
    img.convertTo(img, CV_MAKETYPE(CV_8UC3, channels));

    cvtColor(img, img, cv::COLOR_BGR2RGB);

    auto data{std::make_shared<BaseTensor<T>>(Vector({height, width, channels}))};
    for(INDEX_NBR row{0}; row < height; ++row) {
        for(INDEX_NBR col{0}; col < width; ++col) {
            for(INDEX_NBR channel{0}; channel < channels; ++channel) {
                PIXEL pixel{img.at<cv::Vec3b>(
                    static_cast<int>(row),
                    static_cast<int>(col))[static_cast<int>(channel)]};
                T converted{static_cast<T>(pixel)};
                data->at({row, col, channel}) = converted;
            }
        }
    }
    return data;
}