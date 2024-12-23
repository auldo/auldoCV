#include "machine_learning.h"
#include <iostream>

void testDiabetesDataset() {
    std::string path{"../datasets/diabetes.tab.txt"};
    tensor data{read_tab_separated(path)};
    std::cout << data[{3, 2}] << std::endl;
}

int main() {
    std::cout << "auldo CV" << std::endl;

    auto initial{cv::imread("/Users/dominikaulinger/Desktop/test.jpg", cv::IMREAD_COLOR)};
    auto img{read_image("/Users/dominikaulinger/Desktop/test.jpg")};

    bool test{false};
    for(unsigned long row{0}; row < img->get_dimension(0); ++row) {
        for(unsigned long col{0}; col < img->get_dimension(1); ++col) {
            if(test) {
                img->set_pixel(row, col, {0, 0, 0});
            }
            test = !test;
        }
    }

    auto final {write_image(img, initial.type(), "/Users/dominikaulinger/Desktop/test2.jpg")};
}