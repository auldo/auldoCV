#include "optimizer/optimizer.h"

Optimizer::Optimizer(const std::shared_ptr<Layer> &layer, LossFunction loss, const std::shared_ptr<Tensor<double> > &truth, const std::shared_ptr<Tensor<double> > &inputs) : _lossFunction(loss), _layer(layer), _truth(truth), _inputs(inputs) {}

void Optimizer::updateWeights(PRECISE_NBR learningRate) const {
    //Run over all layers to update all weights and biases.
    auto pos{_layer};
    while(pos != nullptr) {
        if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
            for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                    for(auto& weight : fcNeuron->_weights)
                        weight->applyGradient(learningRate);
                    fcNeuron->_bias->applyGradient(learningRate);
                }
            }
        } else if(auto convLayer = std::dynamic_pointer_cast<ConvolutionalLayer>(pos)) {
            for(INDEX_NBR k{0}; k < convLayer->_kernels.size(); ++k) {
                for(INDEX_NBR r{0}; r < convLayer->_kernels.at(k)->_weights->shapeSize(0); ++r) {
                    for(INDEX_NBR c{0}; c < convLayer->_kernels.at(k)->_weights->shapeSize(1); ++c) {
                        for(INDEX_NBR ch{0}; ch < convLayer->_kernels.at(k)->_weights->shapeSize(2); ++ch) {
                            convLayer->_kernels.at(k)->_weights->at({r, c, ch})->applyGradient(learningRate);
                        }
                    }
                }
            }
        }
        pos = pos->_previous;
    }
}

void Optimizer::updateAverageWeights(PRECISE_NBR learningRate) const {
    //Run over all layers to update all weights and biases.
    auto pos{_layer};
    while(pos != nullptr) {
        if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
            for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                    for(auto& weight : fcNeuron->_weights)
                        weight->applyAverageGradient(learningRate);
                    fcNeuron->_bias->applyAverageGradient(learningRate);
                }
            }
        } else if(auto convLayer = std::dynamic_pointer_cast<ConvolutionalLayer>(pos)) {
            for(INDEX_NBR k{0}; k < convLayer->_kernels.size(); ++k) {
                for(INDEX_NBR r{0}; r < convLayer->_kernels.at(k)->_weights->shapeSize(0); ++r) {
                    for(INDEX_NBR c{0}; c < convLayer->_kernels.at(k)->_weights->shapeSize(1); ++c) {
                        for(INDEX_NBR ch{0}; ch < convLayer->_kernels.at(k)->_weights->shapeSize(2); ++ch) {
                            convLayer->_kernels.at(k)->_weights->at({r, c, ch})->applyAverageGradient(learningRate);
                        }
                    }
                }
            }
        }
        pos = pos->_previous;
    }
}
void Optimizer::rescaleGradientStorages(INDEX_NBR size) const {
    auto pos{_layer};
    while(pos != nullptr) {
        if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
            for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                    for(auto& weight : fcNeuron->_weights)
                        weight->rescaleGradientStorage(size);
                    fcNeuron->_bias->rescaleGradientStorage(size);
                }
            }
        } else if(auto convLayer = std::dynamic_pointer_cast<ConvolutionalLayer>(pos)) {
            for(INDEX_NBR k{0}; k < convLayer->_kernels.size(); ++k) {
                for(INDEX_NBR r{0}; r < convLayer->_kernels.at(k)->_weights->shapeSize(0); ++r) {
                    for(INDEX_NBR c{0}; c < convLayer->_kernels.at(k)->_weights->shapeSize(1); ++c) {
                        for(INDEX_NBR ch{0}; ch < convLayer->_kernels.at(k)->_weights->shapeSize(2); ++ch) {
                            convLayer->_kernels.at(k)->_weights->at({r, c, ch})->rescaleGradientStorage(size);
                        }
                    }
                }
            }
        }
        pos = pos->_previous;
    }
}

void Optimizer::setGradientStorage(INDEX_NBR idx) const {
    auto pos{_layer};
    while(pos != nullptr) {
        if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
            for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                    for(auto& weight : fcNeuron->_weights)
                        weight->setGradientStorage(idx);
                    fcNeuron->_bias->setGradientStorage(idx);
                }
            }
        } else if(auto convLayer = std::dynamic_pointer_cast<ConvolutionalLayer>(pos)) {
            for(INDEX_NBR k{0}; k < convLayer->_kernels.size(); ++k) {
                for(INDEX_NBR r{0}; r < convLayer->_kernels.at(k)->_weights->shapeSize(0); ++r) {
                    for(INDEX_NBR c{0}; c < convLayer->_kernels.at(k)->_weights->shapeSize(1); ++c) {
                        for(INDEX_NBR ch{0}; ch < convLayer->_kernels.at(k)->_weights->shapeSize(2); ++ch) {
                            convLayer->_kernels.at(k)->_weights->at({r, c, ch})->setGradientStorage(idx);
                        }
                    }
                }
            }
        }
        pos = pos->_previous;
    }
}

void Optimizer::updateAverageCloneWeights(PRECISE_NBR learningRate) const {
    //Run over all layers to update all weights and biases.
    auto pos{_layer};
    while(pos != nullptr) {
        if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
            for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                    for(auto& weight : fcNeuron->_weights)
                        weight->applyAverageCloneGradient(learningRate);
                    fcNeuron->_bias->applyAverageCloneGradient(learningRate);
                }
            }
        } else if(auto convLayer = std::dynamic_pointer_cast<ConvolutionalLayer>(pos)) {
            for(INDEX_NBR k{0}; k < convLayer->_kernels.size(); ++k) {
                for(INDEX_NBR r{0}; r < convLayer->_kernels.at(k)->_weights->shapeSize(0); ++r) {
                    for(INDEX_NBR c{0}; c < convLayer->_kernels.at(k)->_weights->shapeSize(1); ++c) {
                        for(INDEX_NBR ch{0}; ch < convLayer->_kernels.at(k)->_weights->shapeSize(2); ++ch) {
                            convLayer->_kernels.at(k)->_weights->at({r, c, ch})->applyAverageCloneGradient(learningRate);
                        }
                    }
                }
            }
        }
        pos = pos->_previous;
    }
}

