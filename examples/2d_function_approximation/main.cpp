#include <auldoCV.h>

int main() {
    auto inputs(std::make_shared<BaseTensor<PRECISE_NBR>>(Vector<INDEX_NBR>({10, 1})));
    inputs->at({0, 0}) = 0.;
    inputs->at({1, 0}) = 1.;
    inputs->at({2, 0}) = 2.;
    inputs->at({3, 0}) = 3.;
    inputs->at({4, 0}) = 4.;
    inputs->at({5, 0}) = 5;
    inputs->at({6, 0}) = 6.;
    inputs->at({7, 0}) = 7.;
    inputs->at({8, 0}) = 8.;
    inputs->at({9, 0}) = 9.;

    auto outputs(std::make_shared<BaseTensor<PRECISE_NBR>>(Vector<INDEX_NBR>({10, 1})));
    outputs->at({0, 0}) = 1.;
    outputs->at({1, 0}) = 0.;
    outputs->at({2, 0}) = 0.;
    outputs->at({3, 0}) = 0.;
    outputs->at({4, 0}) = 0.;
    outputs->at({5, 0}) = 0;
    outputs->at({6, 0}) = 1.;
    outputs->at({7, 0}) = 1.;
    outputs->at({8, 0}) = 1.;
    outputs->at({9, 0}) = 1.;

    auto layer1{std::make_shared<FCLayer>(1, LINEAR, 1)};
    auto layer2{std::make_shared<FCLayer>(4, RELU, layer1)};
    auto layer3{std::make_shared<FCLayer>(4, SIGMOID, layer2)};
    auto layer4{std::make_shared<FCLayer>(1, SIGMOID, layer3)};

    bool sgd{false};
    if(sgd) {
        auto optimizer{std::make_shared<SgdOptimizer>(layer4, 10000, BINARY_CROSS_ENTROPY, outputs, inputs)};
        optimizer->optimize(0.001);
    } else {
        auto optimizer{std::make_shared<ParallelMiniBatchOptimizer>(layer4, 50000, 1, BINARY_CROSS_ENTROPY, outputs, inputs)};
        //optimizer->optimize(0.03);
        optimizer->optimize(0.02);
    }

    for(INDEX_NBR i{0}; i < 10; ++i) {
        layer1->_inputs.value().at(0)->setScalarValue(inputs->at({i, 0}));
        std::cout << "for input " << i << " output is " << layer4->_neurons.at(0)->_output_node->forwardPass() << " should be " << outputs->at({i, 0}) << std::endl;
    }
}