#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "auldoCV.h"

#define RUN_FAST_TESTS_ONLY true

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
    if(!RUN_FAST_TESTS_ONLY) {
        auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
        writeImage("/Users/dominikaulinger/Desktop/test.png", img);
    }
}

TEST_CASE("base kernel test") {
    if(!RUN_FAST_TESTS_ONLY) {
        BaseKernel k{1, 3, 10};
        auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
        auto padding{convertPixelsToByte(k.applyPadding(img))};
        writeImage("/Users/dominikaulinger/Desktop/padding.png", padding);
    }
}

TEST_CASE("simple gaussian blur kernel test") {
    if(!RUN_FAST_TESTS_ONLY) {
        auto img{readImage("/Users/dominikaulinger/Desktop/test.jpg")};
        auto converted{convertPixelsToPrecise(img)};
        auto pixels{convertPixelsToByte(converted)};
        writeImage("/Users/dominikaulinger/Desktop/converted.png", pixels);
    }
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
    if(!RUN_FAST_TESTS_ONLY) {
        auto img{readImage<PRECISE_NBR>("/Users/dominikaulinger/Desktop/test.jpg")};
        auto kernel{SimpleKernel::gaussianBlur()};
        auto result{kernel->apply(img)};
        writeImage("/Users/dominikaulinger/Desktop/gaussian.png", result);
    }
}

TEST_CASE("fcneuron") {
    Vector<PRECISE_NBR> inputs{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Vector<std::shared_ptr<ComputeNode>> nodes(inputs.size());
    for(INDEX_NBR i{0}; i < nodes.size(); ++i)
        nodes.at(i) = COMPUTE_NODE(inputs.at(i));
    CHECK_EQ(nodes.at(5)->getScalarValue(), 6);
    auto neuron{std::make_shared<FCNeuron>(LINEAR, nodes)};
    CHECK_NE(nullptr, neuron->_output_node);
    CHECK_EQ(neuron->_output_node->forwardPass(), 0);
    neuron->_bias->setScalarValue(3);
    CHECK_EQ(neuron->_output_node->forwardPass(), 3);

    auto neuron2{std::make_shared<FCNeuron>(SIGMOID, nodes)};
    neuron2->_bias->setScalarValue(3);
    CHECK_EQ(neuron2->_output_node->forwardPass(), sigmoid(3));
}

TEST_CASE("fc layer") {
    auto input0{Vector({COMPUTE_NODE(0)})};
    auto input1{Vector({COMPUTE_NODE(1)})};
    auto input2{Vector({COMPUTE_NODE(2)})};
    auto input3{Vector({COMPUTE_NODE(3)})};
    auto output{Vector<PRECISE_NBR>({1, 2, 3, 4})};

    auto layer{std::make_shared<FCLayer>(4, SIGMOID, input0)};
    CHECK_EQ(layer->_neurons.size(), 4);

    auto layer2{std::make_shared<FCLayer>(1, SIGMOID, layer)};
    CHECK_EQ(layer2->_neurons.size(), 1);

    auto loss{MSELoss(output.at(0), layer2)};
    CHECK_NE(nullptr, loss._output_node);
}