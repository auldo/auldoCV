#include "vision/simple_kernel.h"

SimpleKernel::SimpleKernel(const Vector<PRECISE_NBR>& filter, PIXEL stride, PIXEL size, std::optional<PIXEL> padding = std::nullopt): BaseKernel(stride, size, padding) {
    if(filter.size() != size*size)
        throw std::runtime_error("input filter data must be of length size ^ 2");
    _filter = std::make_shared<BaseTensor<PRECISE_NBR>>(Vector<INDEX_NBR>({size, size}), filter);
}

std::shared_ptr<SimpleKernel> SimpleKernel::gaussianBlur() {
    auto result{std::make_shared<SimpleKernel>(Vector({
        1./16, 2./16, 1./16,
        2./16, 4./16, 2./16,
        1./16, 2./16, 1./16
    }), 1, 3)};
    return result;
}

std::shared_ptr<SimpleKernel> SimpleKernel::sharpen() {
    auto result{std::make_shared<SimpleKernel>(Vector({
        0., -1., 0.,
        -1., 5., -1.,
        0., -1., 0.,
    }), 1, 3)};
    return result;
}

std::shared_ptr<SimpleKernel> SimpleKernel::edgeDetection() {
    auto result{std::make_shared<SimpleKernel>(Vector({
        -1., -1., -1.,
        -1., 8., -1.,
        -1., -1., -1.,
    }), 1, 3)};
    return result;
}

std::shared_ptr<SimpleKernel> SimpleKernel::boxMean() {
    Vector<PRECISE_NBR> filter(9);
    filter.fill(1./9);
    auto result{std::make_shared<SimpleKernel>(filter, 1, 3)};
    return result;
}

std::shared_ptr<Tensor<PRECISE_NBR>> SimpleKernel::apply(TENSOR_REF(PRECISE_NBR)& input) {
    INDEX_NBR newHeight{this->calculateOutputDimension(input->shapeSize(0))};
    INDEX_NBR newWidth{this->calculateOutputDimension(input->shapeSize(1))};

    auto img{std::make_shared<BaseTensor<PRECISE_NBR>>(Vector({newHeight, newWidth, input->shapeSize(2)}))};

    //For each channel of the input image
    for(INDEX_NBR channel{0}; channel < input->shapeSize(2); ++channel) {
        //Iterate over every pixel of new image
        for(INDEX_NBR row{0}; row < newHeight; ++row) {
            for(INDEX_NBR col{0}; col < newWidth; ++col) {
                INDEX_NBR startX{row * this->_stride};
                INDEX_NBR startY{col * this->_stride};
                PRECISE_NBR sum{0};
                for(INDEX_NBR x{0}; x < _size; ++x) {
                    for(INDEX_NBR y{0}; y < _size; ++y) {
                        sum += TO_PRECISE_NBR(input->at({x + startX, y + startY, channel})) * _filter->at({x, y});
                    }
                }
                img->at({row, col, channel}) = sum;
            }
        }
    }

    return img;
}
