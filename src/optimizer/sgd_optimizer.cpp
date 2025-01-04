#include "optimizer/sgd_optimizer.h"

#include <data/nested_tensor.h>
#include <layer/fc_layer.h>
#include <loss/binary_cross_entropy_loss.h>
#include <loss/mse_loss.h>
#include <neuron/fc_neuron.h>

SgdOptimizer::SgdOptimizer(const std::shared_ptr<Layer>& finalLayer, INDEX_NBR epochs, LossFunction loss, const std::shared_ptr<Tensor<PRECISE_NBR>>& truth, const std::shared_ptr<Tensor<PRECISE_NBR>>& inputs) : _epochs(epochs), _lossFunction(loss), _truth(truth), _layer(finalLayer), _inputs(inputs) {
    if(truth->rank() != 2 && truth->rank() != 1)
        throw std::runtime_error("truth must be of rank 1 or 2");
    if(inputs->rank() != 2 && inputs->rank() != 1)
        throw std::runtime_error("truth must be of rank 1 or 2");
    if(truth->shapeSize(0) != inputs->shapeSize(0))
        throw std::runtime_error("truth and input must be of same size at dim 0");
}

void SgdOptimizer::optimize(PRECISE_NBR learningRate) {
    auto firstLayer{_layer};
    while(firstLayer->_previous != nullptr)
        firstLayer = firstLayer->_previous;
    learningRate = learningRate * _truth->shapeSize(0);

    std::cout << "starting first epoch";

    //Run epochs
    for(auto e{0}; e < _epochs; ++e) {

        PRECISE_NBR epochLoss{0};

        //Run iteration with a specific in- and output.
        for(auto i{0}; i < _truth->shapeSize(0); ++i) {

            //Unpack truth for this iteration.
            auto iterationTruth{_truth->operator[](i)};

            //Unpack input for this iteration and store it in a vector.
            auto iterationInput{_inputs->operator[](i)};
            Vector<PRECISE_NBR> input(iterationInput->rank() == 0 ? 1 : iterationInput->shapeSize(0));
            if(iterationInput->rank() == 0) {
                input.at(0) = iterationInput->scalar();
            } else {
                for(INDEX_NBR in{0}; in < iterationInput->shapeSize(0); ++in)
                    input.at(in) = iterationInput->at({in});
            }

            //Set input into the very first layer compute input nodes.
            if(auto layer = std::dynamic_pointer_cast<FCLayer>(firstLayer)) {
                for(auto in{0}; in < layer->_inputs.value().size(); ++in) {
                    //std::cout << "setting value to " << input.at(in) << std::endl;
                    layer->_inputs.value().at(in)->setScalarValue(input.at(in));
                }
            }

            //Append loss to the compute graph.
            Loss loss;
            switch(_lossFunction) {
                case MSE:
                    loss = MSELoss(iterationTruth->scalar(), _layer);
                    break;
                case BINARY_CROSS_ENTROPY:
                    loss = BinaryCrossEntropyLoss(iterationTruth->scalar(), _layer);
                    break;
            }

            //Execute forward and backward passes of the compute graph to derive gradients.
            epochLoss += loss._output_node->forwardPass();
            loss._output_node->backwardPass();

            //Run over all layers to update all weights and biases.
            auto pos{_layer};
            while(pos != nullptr) {
                if(auto fcLayer = std::dynamic_pointer_cast<FCLayer>(pos)) {
                    for(INDEX_NBR n{0}; n < fcLayer->_neurons.size(); ++n) {
                        if(auto fcNeuron = std::dynamic_pointer_cast<FCNeuron>(fcLayer->_neurons.at(n))) {
                            for(auto& weight : fcNeuron->_weights)
                                weight->setScalarValue(weight->getScalarValue() - learningRate * weight->gradient());
                            fcNeuron->_bias->setScalarValue(fcNeuron->_bias->getScalarValue() - learningRate * fcNeuron->_bias->gradient());
                        }
                    }
                }
                pos = pos->_previous;
            }
        }

        epochLoss /= _truth->shapeSize(0);

        std::cout<<"\r \r";
        std::cout << "epoch done, avg loss: " << epochLoss;
    }

    std::cout << std::endl;
}

