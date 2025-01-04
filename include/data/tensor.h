#pragma once
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <gradient/compute_node.h>

#include "data/linearized.h"
#include "data/vector.h"

template <typename TENSOR_TYPE>
class NestedTensor;

template <typename TENSOR_TYPE>
class Tensor : public Linearized {
protected:
    Tensor(): Linearized() {};
public:
    /// Uses _transform_indices to access element at certain index and returns its reference.
    /// May throw out of range.
    virtual TENSOR_TYPE& at(Vector<INDEX_NBR> indices) = 0;

    /// Terminology
    /// A scalar is represented as [value] and has rank 0.
    /// An array is of rank 1.
    /// A matrix (e.g., 4x3 is rank 2).
    USE_RETURN virtual INDEX_NBR rank() const = 0;
    virtual void assertRank(INDEX_NBR rank) const = 0;

    /// Interprets tensor as scalar.
    /// Fails if tensor isn't a scalar (hasn't rank 0 conditions fulfilled).
    USE_RETURN virtual TENSOR_TYPE& scalar() = 0;

    /// Resizes the vector to other dimensions.
    /// Only works if new size can be transferred into the same size of linearized array.
    virtual void resize(const Vector<INDEX_NBR>& dimensions) = 0;

    virtual std::shared_ptr<NestedTensor<TENSOR_TYPE>> operator[](INDEX_NBR index) = 0;
};