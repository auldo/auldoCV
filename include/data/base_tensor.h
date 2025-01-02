#pragma once
#include <stdexcept>
#include <string>

#include "data/nested_tensor.h"
#include "data/tensor.h"
#include "data/vector.h"

template <typename TENSOR_TYPE>
class BaseTensor : public Tensor<TENSOR_TYPE>, public std::enable_shared_from_this<BaseTensor<TENSOR_TYPE>> {

    /// The linearized data.
    Vector<TENSOR_TYPE> _data{0};

public:

    /// Creates a tensor of a certain size.
    explicit BaseTensor(const Vector<INDEX_NBR> &dimensions) {
        this->_dimensions = dimensions;
        _data.resize(this->_dimensions.multiplied_sum());
    }

    BaseTensor(const Vector<INDEX_NBR>& dimensions, const Vector<TENSOR_TYPE>& data) {
        this->_dimensions = dimensions;
        this->_data = data;
    }

    /// Creates a scalar tensor i.e., a tensor having rank 0 and one dimension of length 1 with one element.
    explicit BaseTensor(const TENSOR_TYPE& scalar): _data({scalar}) {
        this->_dimensions = Vector<INDEX_NBR>({1});
    }

    /// Uses _transform_indices to access element at certain index and returns its reference.
    /// May throw out of range.
    TENSOR_TYPE& at(Vector<INDEX_NBR> indices) override {
        if(indices.size() != this->_dimensions.size())
            throw std::invalid_argument("expected "  + std::to_string(this->_dimensions.size()) + " indices.");
        for(auto i{0}; i < this->_dimensions.size(); ++i) {
            if(indices.at(i) >= this->_dimensions.at(i))
                throw std::out_of_range("index out of range");
        }
        auto index{this->_transform_indices(indices)};
        return _data.at(index);
    }

    /// Terminology
    /// A scalar is represented as [value] and has rank 0.
    /// An array is of rank 1.
    /// A matrix (e.g., 4x3 is rank 2).
    USE_RETURN INDEX_NBR rank() const override {
        if(this->_dimensions.size() == 1 && _data.size() == 1)
            return 0;
        INDEX_NBR length1Dimensions{0};
        for(const auto& elem : this->_dimensions)
            length1Dimensions += elem == 1 ? elem : 0;
        return this->_dimensions.size() - length1Dimensions;
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

    std::shared_ptr<NestedTensor<TENSOR_TYPE>> operator[](INDEX_NBR index) override {
        Vector<INDEX_NBR> updatedDimensions(this->_dimensions.size() - 1);
        for(auto i{0}; i < updatedDimensions.size(); ++i)
            updatedDimensions.at(i) = this->_dimensions.at(i + 1);
        auto nested{std::make_shared<NestedTensor<TENSOR_TYPE>>(updatedDimensions, this->shared_from_this(), index)};
        return nested;
    };
};