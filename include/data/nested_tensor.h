#pragma once
#include <stdexcept>

#include "data/vector.h"
#include "data/tensor.h"

template <typename TENSOR_TYPE>
class NestedTensor : public Tensor<TENSOR_TYPE>, public std::enable_shared_from_this<NestedTensor<TENSOR_TYPE>> {

    std::shared_ptr<Tensor<TENSOR_TYPE>> _parent;
    INDEX_NBR _index;

public:

    /// Creates a tensor of a certain size.
    explicit NestedTensor(const Vector<INDEX_NBR>& dimensions, const std::shared_ptr<Tensor<TENSOR_TYPE>>& parent, const INDEX_NBR index): _parent(parent), _index(index) {
        this->_dimensions = dimensions;
    }

    /// Uses _transform_indices to access element at certain index and returns its reference.
    /// May throw out of range.
    TENSOR_TYPE& at(Vector<INDEX_NBR> indices) override {
        Vector<INDEX_NBR> accessIndices(indices.size() + 1);
        accessIndices.at(0) = _index;
        for(INDEX_NBR i{1}; i < accessIndices.size(); ++i) {
            accessIndices.at(i) = indices.at(i - 1);
        }
        return _parent->at(accessIndices);
    }

    /// Terminology
    /// A scalar is represented as [value] and has rank 0.
    /// An array is of rank 1.
    /// A matrix (e.g., 4x3 is rank 2).
    USE_RETURN INDEX_NBR rank() const override {
        if(this->_dimensions.size() == 1 /*_parent->_data.size() == 1*/)
            return 0;
        return this->_dimensions.size();
    }

    /// Interprets tensor as scalar.
    /// Fails if tensor isn't a scalar (hasn't rank 0 conditions fulfilled).
    USE_RETURN TENSOR_TYPE& scalar() override {
        if(rank() != 0)
            throw std::invalid_argument("rank must be 0");
        return this->at({0});
    }

    /// Resizes the vector to other dimensions.
    /// Only works if new size can be transferred into the same size of linearized array.
    void resize(const Vector<INDEX_NBR>& dimensions) override {
        if(this->_dimensions.multiplied_sum() != dimensions.multiplied_sum())
            throw std::invalid_argument("can't resize tensor");
        this->_dimensions = dimensions;
    }

    void assertRank(INDEX_NBR rank) const override {
        if(this->rank() != rank)
            throw std::runtime_error("bad rank");
    }

    std::shared_ptr<NestedTensor> operator[](INDEX_NBR index) {
        Vector<INDEX_NBR> updatedDimensions(this->_dimensions.size() - 1);
        for(auto i{0}; i < updatedDimensions.size(); ++i)
            updatedDimensions.at(i) = this->_dimensions.at(i);
        auto nested{std::make_shared<NestedTensor>(updatedDimensions, this->shared_from_this(), index)};
        return nested;
    };
};