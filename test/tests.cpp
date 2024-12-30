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

TEST_CASE("read cifar batch") {
    readCifar10Batch("/Users/dominikaulinger/Desktop/cifar10", CIFAR_10_BATCH_2);
}