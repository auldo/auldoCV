#include "optimizer/sgd_optimizer.h"

#include <data/nested_tensor.h>
#include <layer/conv_layer.h>
#include <layer/fc_layer.h>
#include <loss/binary_cross_entropy_loss.h>
#include <loss/cross_entropy_loss.h>
#include <loss/mse_loss.h>
#include <neuron/fc_neuron.h>

SgdOptimizer::SgdOptimizer(const std::shared_ptr<Layer>& finalLayer, INDEX_NBR epochs, LossFunction loss, const std::shared_ptr<Tensor<PRECISE_NBR>>& truth, const std::shared_ptr<Tensor<PRECISE_NBR>>& inputs) : Optimizer(finalLayer, loss, truth, inputs), _epochs(epochs) {
    if(truth->rank() != 2 && truth->rank() != 1)
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

            //Set input into the very first layer compute input nodes.
            if(auto layer = std::dynamic_pointer_cast<FCLayer>(firstLayer)) {
                //Unpack input for this iteration and store it in a vector.
                auto iterationInput{_inputs->operator[](i)};
                Vector<PRECISE_NBR> input(iterationInput->rank() == 0 ? 1 : iterationInput->shapeSize(0));
                if(iterationInput->rank() == 0) {
                    input.at(0) = iterationInput->scalar();
                } else {
                    for(INDEX_NBR in{0}; in < iterationInput->shapeSize(0); ++in)
                        input.at(in) = iterationInput->at({in});
                }

                for(auto in{0}; in < layer->_inputs.value().size(); ++in) {
                    //std::cout << "setting value to " << input.at(in) << std::endl;
                    layer->_inputs.value().at(in)->setScalarValue(input.at(in));
                }
            }

            if(auto layer = std::dynamic_pointer_cast<ConvolutionalLayer>(firstLayer)) {
                auto iterationInput{_inputs->operator[](i)};
                layer->setInputs(iterationInput);
            }

            //Append loss to the compute graph.
            Loss loss;
            switch(_lossFunction) {
                case MSE:
                    loss = MSELoss(iterationTruth->scalar(), std::dynamic_pointer_cast<FCLayer>(_layer)->_neurons.at(0)->_output_node);
                    break;
                case BINARY_CROSS_ENTROPY:
                    loss = BinaryCrossEntropyLoss(iterationTruth->scalar(), std::dynamic_pointer_cast<FCLayer>(_layer)->_neurons.at(0)->_output_node);
                    break;
                case CROSS_ENTROPY:
                    loss = CrossEntropyLoss(iterationTruth->scalar(), std::dynamic_pointer_cast<FCLayer>(_layer)->getComputeNodes());
            }

            //Execute forward and backward passes of the compute graph to derive gradients.
            const PRECISE_NBR iterationLoss{loss._output_node->forwardPass()};
            epochLoss += iterationLoss;
            loss._output_node->backwardPass();

            //Run over all layers to update all weights and biases.
            updateWeights(learningRate);
        }

        epochLoss /= _truth->shapeSize(0);

        std::cout<<"\r \r";
        std::cout << "epoch done, avg loss: " << epochLoss;
    }

    std::cout << std::endl;
}

