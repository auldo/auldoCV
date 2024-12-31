#pragma once
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>

#include "vector.h"

template <typename TENSOR_TYPE>
class Tensor {
    // Iterator types
    using tensor_iterator = TENSOR_TYPE*;
    using const_tensor_iterator = const TENSOR_TYPE *;

    /// The linearized data.
    Vector<TENSOR_TYPE> _data{0};

    /// The dimensionality of the tensor.
    /// E.g., a matrix with 3 rows and 4 columns would have the value {3, 4}.
    Vector<INDEX_NBR> _dimensions{0};

public:
    /**
    * Answers the question: When storing an n-dimensional array linearized (i.e. in a one-dimensional array), which index do we look up for the set of indices { i1, i2, ..., in }?
    * In the following examples, Dimx (e.g., Dim1, Dim2) means "requested index at dimension x".
    * Dimx can be found in the parameter indices.
    * In the following examples, len(Dimx) means "size of dimension x".
    * len(Dimx) can be found in the instance variable _dimensions.
    * Example for two dimensions: Idx = len(Dim2) * Dim1 + Dim2
    * Example for three dimensions: Idx = Dim1 * len(Dim2) * len(Dim3) + Dim2 * len(Dim3) + Dim3
    * This method formalizes the two examples for n >= 0 dimensions.
    */
    USE_RETURN INDEX_NBR _transform_indices(Vector<INDEX_NBR>& indices) const {
        INDEX_NBR index{0};
        for(auto i{0}; i < indices.size(); ++i) {
            auto idx{indices.size() - i - 1};
            auto sum{1};
            for(auto i2{idx+1}; i2 < indices.size(); ++i2) {
                sum *= _dimensions.at(i2);
            }
            index += (indices.at(idx) * sum);
        }
        return index;
    }

    /// Does the opposite as compared to _transform_indices.
    /// Transforms an index referencing the linearized array to an index usable in a multidimensional array.
    USE_RETURN Vector<INDEX_NBR> _transform_index(INDEX_NBR index) const {
        if(index > max_index())
            throw std::out_of_range("index out of range");
        Vector<INDEX_NBR> result(_dimensions.size());
        INDEX_NBR last_n{_dimensions.size() - 1}; //3
        while(last_n > 0) {
            INDEX_NBR prod{_dimensions.multiplied_sum_last_n(last_n)};
            INDEX_NBR divisor{index / prod};
            index = index - prod * divisor;
            result.at(_dimensions.size() - 1 - last_n) = divisor;
            --last_n;
        }
        result.at(_dimensions.size() - 1) = index % _dimensions.at(_dimensions.size() - 1);
        return result;
    }

    /// Returns the maximum possible index of the linearized array.
    USE_RETURN INDEX_NBR max_index() const {
        return _dimensions.multiplied_sum() - 1;
    }

public:

    /// Creates a tensor of a certain size.
    explicit Tensor(Vector<INDEX_NBR> dimensions) {
        _dimensions = std::move(dimensions);
        _data.resize(_dimensions.multiplied_sum());
    }

    /// Creates a scalar tensor i.e., a tensor having rank 0 and one dimension of length 1 with one element.
    explicit Tensor(const TENSOR_TYPE& scalar): _dimensions({1}), _data({scalar}) {}

    /// Creates empty vector of size zero.
    Tensor(): _data(), _dimensions() {}

    USE_RETURN INDEX_NBR shapeSize(unsigned index) const {
        return _dimensions.at(index);
    }

    /// Copy-assigns a scalar to this vector leading to a rank 0 tensor with one element.
    Tensor& operator=(const TENSOR_TYPE& scalar) {
        this->_dimensions = {1};
        this->_data = {scalar};
        return *this;
    }

    /// Uses _transform_indices to access element at certain index and returns its reference.
    /// May throw out of range.
    TENSOR_TYPE& at(Vector<INDEX_NBR> indices) {
        if(indices.size() != _dimensions.size())
            throw std::invalid_argument("expected "  + std::to_string(_dimensions.size()) + " indices.");
        for(auto i{0}; i < _dimensions.size(); ++i) {
            if(indices.at(i) >= _dimensions.at(i))
                throw std::out_of_range("index out of range");
        }
        auto index{_transform_indices(indices)};
        return _data.at(index);
    }

    /// Terminology
    /// A scalar is represented as [value] and has rank 0.
    /// An array is of rank 1.
    /// A matrix (e.g., 4x3 is rank 2).
    USE_RETURN INDEX_NBR rank() const {
        if(_dimensions.size() == 1 && _data.size() == 1)
            return 0;
        return _dimensions.size();
    }

    /// Interprets tensor as scalar.
    /// Fails if tensor isn't a scalar (hasn't rank 0 conditions fulfilled).
    USE_RETURN TENSOR_TYPE& scalar() {
        if(rank() != 0)
            throw std::invalid_argument("rank must be 0");
        return this->at({0});
    }

    /// Resizes the vector to other dimensions.
    /// Only works if new size can be transferred into the same size of linearized array.
    void resize(Vector<INDEX_NBR>&& dimensions) {
        if(_dimensions.multiplied_sum() != dimensions.multiplied_sum())
            throw std::invalid_argument("can't resize tensor");
        _dimensions = std::move(_dimensions);
    }

    // Iterators over the linearized array.
    USE_RETURN tensor_iterator begin() { return _data.begin(); }
    USE_RETURN tensor_iterator end() { return _data.end(); }

    // Const iterators over the linearized arrays.
    USE_RETURN const_tensor_iterator begin() const { return _data.begin(); }
    USE_RETURN const_tensor_iterator end() const { return _data.end(); }

    void assertRank(INDEX_NBR rank) const {
        if(this->rank() != rank)
            throw std::runtime_error("bad rank");
    }

    std::shared_ptr<Tensor> operator[](INDEX_NBR index) {
        auto tensor{std::make_shared<Tensor>()};
        tensor->_dimensions.resize(_dimensions.size() - 1);
        for(INDEX_NBR i{1}; i < _dimensions.size(); ++i)
            tensor->_dimensions.at(i - 1) = _dimensions.at(i);

        Vector<INDEX_NBR> firstPosIndices(_dimensions.size(), 0);
        Vector<INDEX_NBR> lastPosIndices(_dimensions.size(), 0);

        firstPosIndices.at(0) = index;
        lastPosIndices.at(0) = index + 1;

        auto firstPos{_transform_indices(firstPosIndices)};
        auto lastPos{_transform_indices(lastPosIndices) - 1};

        tensor->_data = Vector(_data, std::make_pair(firstPos, lastPos));
        return tensor;
    };
};