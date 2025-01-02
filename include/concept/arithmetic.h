#pragma once

#include <utility>

template <typename T>
/// A concept allowing for integral and floating point (so in general arithmetical numerical) values.
concept arithmetic = std::integral<T> or std::floating_point<T>;