#pragma once

class Optimizer {
protected:
    Optimizer() = default;
public:
    virtual ~Optimizer() = default;
    virtual void optimize(PRECISE_NBR learningRate) = 0;
};