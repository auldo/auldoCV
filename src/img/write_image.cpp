#include "img/write_image.h"

void writeImage(const std::string& path, const TENSOR_REF(PIXEL)& data) {
    INDEX_NBR height{data->shapeSize(0)};
    INDEX_NBR width{data->shapeSize(1)};
    INDEX_NBR channels{data->shapeSize(2)};
    cv::Mat img(cv::Size(static_cast<int>(width), static_cast<int>(height)), CV_8UC3);

    for(INDEX_NBR r{0}; r < img.rows; ++r) {
        for(INDEX_NBR c{0}; c < img.cols; ++c) {
            for(INDEX_NBR channel{0}; channel < channels; ++channel) {
                PIXEL value{data->at({r, c, channel})};
                img.at<cv::Vec3b>(
                    static_cast<int>(r),
                    static_cast<int>(c))[static_cast<int>(channel)] = value;
            }
        }
    }

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img);
}