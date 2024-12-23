#include "write/write_image.h"

std::shared_ptr<cv::Mat> write_image(const std::shared_ptr<tensor<int>>& data, int type, const std::string& filename) {
    auto channelCount{data->get_dimension(2)};

    auto height{data->get_dimension(0)};
    auto width{data->get_dimension(1)};

    auto img{std::make_shared<cv::Mat>(cv::Mat{cv::Size{
        static_cast<int>(width),
        static_cast<int>(height)
    }, type})};

    img->convertTo(*img, type);
    img->convertTo(*img, cv::DataType<unsigned char>::type);

    for(unsigned long r{0}; r < img->rows; ++r) {
        for(unsigned long c{0}; c < img->cols; ++c) {
            for(unsigned long channel{0}; channel < channelCount; ++channel) {
                img->at<cv::Vec3b>(
                    static_cast<int>(r),
                    static_cast<int>(c))[static_cast<int>(channel)] = static_cast<uchar>((*data)[{r, c, channel}]);
            }
        }
    }

    std::cout << "writing image..." << std::endl;
    imwrite(filename, *img);

    return img;
}