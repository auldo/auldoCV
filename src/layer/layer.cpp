#include "layer/layer.h"

Layer::Layer(const std::shared_ptr<Layer>& previous): _previous(previous) {}