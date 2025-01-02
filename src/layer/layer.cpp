#include "layer/layer.h"

Layer::Layer(INDEX_NBR size, const std::shared_ptr<Layer>& previous): _neurons(size), _previous(previous) {}