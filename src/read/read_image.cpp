#include "read/read_image.h"

std::shared_ptr<tensor<int>> read_image(const std::string& image_path) {
    auto img = cv::imread(image_path, cv::IMREAD_COLOR);

    auto width{static_cast<unsigned long>(img.cols)};
    auto height{static_cast<unsigned long>(img.rows)};
    auto channelCount{static_cast<unsigned long>(img.channels())};
    auto data{std::make_shared<tensor<int>>(tensor<int>({
        height, width, channelCount
    }))};
    for(unsigned long row{0}; row < height; ++row) {
        for(unsigned long col{0}; col < width; ++col) {
            for(unsigned long channel{0}; channel < channelCount; ++channel) {
                (*data)[{row, col, channel}] = img.at<cv::Vec3b>(
                    static_cast<int>(row),
                    static_cast<int>(col))[static_cast<int>(channel)];
            }
        }
    }
    return data;
}