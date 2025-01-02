#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "auldoCV.h"

TEST_CASE("basic compute node scalar tests") {
    auto node{COMPUTE_NODE(4)};
    CHECK_EQ(node->getScalarValue(), 4);
    CHECK_THROWS(node->gradient());
}

TEST_CASE("compute node test 1") {
    auto x{COMPUTE_NODE(1)};
    auto w1{COMPUTE_NODE(1./3.)};
    auto w2{COMPUTE_NODE(2)};
    auto y{COMPUTE_NODE(0)};
    auto yNegative{COMPUTE_NODE_TIMES_CONST(y, -1)};
    auto z{COMPUTE_NODE_TIMES(x, w1)};
    auto activation{COMPUTE_NODE_RELU(z)};
    auto estimation{COMPUTE_NODE_TIMES(activation, w2)};
    auto loss{COMPUTE_NODE_PLUS(yNegative, estimation)};
    auto lossSquared{COMPUTE_NODE_POW(loss, 2)};

    for(auto i{0}; i < 2; ++i) {
        CHECK_EQ(lossSquared->forwardPass(), 4./9);
        CHECK_EQ(lossSquared->forwardPassCache.size(), 1);

        lossSquared->backwardPass();
        CHECK_EQ(w1->gradient(), 8./3);
        CHECK_EQ(w2->gradient(), 4./9);
    }
}

TEST_CASE("subtensor") {
    auto t{std::make_shared<BaseTensor<int>>(Vector<INDEX_NBR>({3, 3, 3}))};
    auto subTensor{t->operator[](0)};

    subTensor->at({0, 0}) = 10;
    CHECK_EQ(t->at({0, 0, 0}), 10);
    CHECK_THROWS(subTensor->at({0, 10}));
    CHECK_THROWS(subTensor->at({0}));

    auto subTensor2{subTensor->operator[](0)};
    CHECK_EQ(subTensor2->at({0}), 10);
}

TEST_CASE("read image test") {
    auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
    writeImage("/Users/dominikaulinger/Desktop/test.png", img);
}

TEST_CASE("base kernel test") {
    BaseKernel k{1, 3, 10};
    auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
    auto padding{convertPixelsToByte(k.applyPadding(img))};
    writeImage("/Users/dominikaulinger/Desktop/padding.png", padding);
}

TEST_CASE("simple gaussian blur kernel test") {
    auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
    auto converted{convertPixelsToPrecise(img)};
    auto pixels{convertPixelsToByte(converted)};
    writeImage("/Users/dominikaulinger/Desktop/converted.png", pixels);
}

TEST_CASE("test kernel") {
    auto kernel{SimpleKernel::gaussianBlur()};
    CHECK_EQ(kernel->_filter->at({0, 0}), 1./16);
    CHECK_EQ(kernel->_filter->at({0, 1}), 2./16);
    CHECK_EQ(kernel->_filter->at({0, 2}), 1./16);
    CHECK_EQ(kernel->_filter->at({1, 0}), 2./16);
    CHECK_EQ(kernel->_filter->at({1, 1}), 4./16);
    CHECK_EQ(kernel->_filter->at({1, 2}), 2./16);
    CHECK_EQ(kernel->_filter->at({2, 0}), 1./16);
    CHECK_EQ(kernel->_filter->at({2, 1}), 2./16);
    CHECK_EQ(kernel->_filter->at({2, 2}), 1./16);
}

TEST_CASE("simple gaussian blur kernel test") {
    auto img{readImage<PRECISE_NBR>("/Users/dominikaulinger/Desktop/test.jpg")};
    auto kernel{SimpleKernel::gaussianBlur()};
    auto result{kernel->apply(img)};
    writeImage("/Users/dominikaulinger/Desktop/gaussian.png", result);
}