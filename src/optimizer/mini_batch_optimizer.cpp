#include "optimizer/mini_batch_optimizer.h"

MiniBatchOptimizer::MiniBatchOptimizer(const std::shared_ptr<Layer> &finalLayer, unsigned int iterations, unsigned int miniBatchSize, LossFunction loss, const std::shared_ptr<Tensor<double> > &truth, const std::shared_ptr<Tensor<double> > &inputs) : Optimizer(finalLayer, loss, truth, inputs), _iterations(iterations), _batchSize(miniBatchSize) {}

USE_RETURN Vector<INDEX_NBR> MiniBatchOptimizer::selectMiniBatch() const {
    static std::random_device rdev;
    static std::default_random_engine re(rdev());
    typedef std::conditional_t<
        std::is_floating_point_v<int>,
        std::uniform_real_distribution<int>,
        std::uniform_int_distribution<int>> dist_type;
    dist_type uni(0, static_cast<int>(_inputs->shapeSize(0)) - 1);

    Vector<INDEX_NBR> selection(_batchSize);
    for(INDEX_NBR i{0}; i < _batchSize; ++i)
        selection.at(i) = uni(re);
    return selection;
}

void MiniBatchOptimizer::optimize(double learningRate) {
    auto firstLayer{_layer};
    while(firstLayer->_previous != nullptr)
        firstLayer = firstLayer->_previous;

    std::cout << "starting first iteration";

    rescaleGradientStorages(_batchSize);

    //Run epochs
    for(auto e{0}; e < _iterations; ++e) {

        PRECISE_NBR epochLoss{0};
        auto miniBatchSelection{selectMiniBatch()};

        //Run iteration with a specific in- and output.
        for(auto s{0}; s < miniBatchSelection.size(); ++s) {
            INDEX_NBR i{miniBatchSelection.at(s)};

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
                    loss = MSELoss(iterationTruth->scalar(), _layer);
                    break;
                case BINARY_CROSS_ENTROPY:
                    loss = BinaryCrossEntropyLoss(iterationTruth->scalar(), _layer);
                    break;
                case CROSS_ENTROPY:
                    loss = CrossEntropyLoss(TO_PIXEL(iterationTruth->scalar()), _layer);
            }

            //Execute forward and backward passes of the compute graph to derive gradients.
            const PRECISE_NBR iterationLoss{loss._output_node->forwardPass()};
            epochLoss += iterationLoss;
            loss._output_node->backwardPass();
            setGradientStorage(s);
        }

        //Run over all layers to update all weights and biases.
        updateAverageWeights(learningRate);

        epochLoss /= _truth->shapeSize(0);

        std::cout<<"\r \r";
        std::cout << "mini batch iteration done, avg loss: " << epochLoss;
    }

    std::cout << std::endl;
}
