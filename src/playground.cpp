#include "machine_learning.h"
#include <iostream>

int main() {
    std::cout << "auldo CV" << std::endl;

    auto a{std::make_shared<ComputationalGraphNode>(ComputationalGraphNode(2))};
    auto b{std::make_shared<ComputationalGraphNode>(ComputationalGraphNode(1))};
    auto b2{std::make_shared<ComputationalGraphNode>(ComputationalGraphNode(1))};
    auto c{std::make_shared<ComputationalGraphNode>(OP_PLUS, std::vector({a, b}))};
    auto d{std::make_shared<ComputationalGraphNode>(OP_PLUS, std::vector({b, b2}))};
    auto e{std::make_shared<ComputationalGraphNode>(OP_TIMES, std::vector({c, d}))};

    std::cout << e->forwardPass() << std::endl;
    e->backwardPass(std::optional<float>{});

    std::cout << a->gradient() << std::endl;
    std::cout << b->gradient() << std::endl;
    std::cout << c->gradient() << std::endl;
    std::cout << d->gradient() << std::endl;
    std::cout << e->gradient() << std::endl;

    /*
    std::cout << e->backwardPassCache.at(0) << std::endl;
    std::cout << e->backwardPassCache.at(1) << std::endl;

    std::cout << c->backwardPassCache.at(0) << std::endl;
    std::cout << c->backwardPassCache.at(1) << std::endl;

    std::cout << d->backwardPassCache.at(1) << std::endl;
    */
}