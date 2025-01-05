#include <auldoCV.h>

int main() {
    auto cifar{readCifar10Batch<PRECISE_NBR>("/Users/dominikaulinger/Desktop/cifar-10", CIFAR_10_BATCH_1)};
    std::cout << "cifar batch loaded" << std::endl;

    auto layer1{std::make_shared<ConvolutionalLayer>(RELU, 3, 2, 3, 32, 32, 3)};
    auto layer2{std::make_shared<ConvolutionalLayer>(RELU, layer1, 2, 3, 1)};
    auto layer3{std::make_shared<FCLayer>(10, SIGMOID, layer2)};

    ParallelMiniBatchOptimizer optimizer(layer3, 10, 32, CROSS_ENTROPY, cifar.second, cifar.first);
    optimizer.optimize(0.001);
}