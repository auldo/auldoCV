#ifndef TENSOR_H
#define TENSOR_H

#define USE_RETURN [[nodiscard]]

#include <iostream>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "vector.h"

template <typename tensor_type> requires arithmetic<tensor_type>
class tensor {

    // Iterator types
    using tensor_iterator = tensor_type*;
    using const_tensor_iterator = const tensor_type *;

    /// The linearized data.
    vector<tensor_type> _data{0};

    /// The dimensionality of the tensor.
    /// E.g., a matrix with 3 rows and 4 columns would have the value {3, 4}.
    vector<size_t> _dimensions{0};
public:

    /// Creates a tensor of a certain size.
    explicit tensor(vector<size_t> dimensions) {
        _dimensions = std::move(dimensions);
        _data.reset_size(_dimensions.multiplied_sum());
    }

    /// Creates a scalar tensor i.e., a tensor having rank 0 and one dimension of length 1 with one element.
    explicit tensor(const tensor_type& scalar): _dimensions({1}), _data({scalar}) {}

    USE_RETURN size_t get_dimension(unsigned index) const {
        return _dimensions[index];
    }

    /// Copy-assigns a scalar to this vector leading to a rank 0 tensor with one element.
    tensor& operator=(const tensor_type& scalar) {
        this->_dimensions = {1};
        this->_data = {scalar};
        return *this;
    }

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
    USE_RETURN size_t _transform_indices(vector<size_t>& indices) const {
        size_t index{0};
        for(auto i{0}; i < indices.size(); ++i) {
            auto idx{indices.size() - i - 1};
            auto sum{1};
            for(auto i2{idx+1}; i2 < indices.size(); ++i2) {
                sum *= _dimensions[i2];
            }
            index += (indices[idx] * sum);
        }
        return index;
    }

    /// Does the opposite as compared to _transform_indices.
    /// Transforms an index referencing the linearized array to an index usable in a multidimensional array.
    USE_RETURN vector<size_t> _transform_index(size_t index) const {
        if(index > max_index())
            throw std::out_of_range("index out of range");
        vector<size_t> result(_dimensions.size());
        size_t last_n{_dimensions.size() - 1}; //3
        while(last_n > 0) {
            size_t prod{_dimensions.multiplied_sum_last_n(last_n)};
            size_t divisor{index / prod};
            index = index - prod * divisor;
            result[_dimensions.size() - 1 - last_n] = divisor;
            --last_n;
        }
        result[_dimensions.size() - 1] = index % _dimensions[_dimensions.size() - 1];
        return result;
    }

    /// Uses _transform_indices to access element at certain index and returns its reference.
    /// May throw out of range.
    tensor_type& operator[](vector<size_t> indices) {
        if(indices.size() != _dimensions.size())
            throw std::invalid_argument("expected "  + std::to_string(_dimensions.size()) + " indices.");
        for(auto i{0}; i < _dimensions.size(); ++i) {
            if(indices[i] >= _dimensions[i])
                throw std::out_of_range("index out of range");
        }
        auto index{_transform_indices(indices)};
        return _data[index];
    }

    /// Terminology
    /// A scalar is represented as [value] and has rank 0.
    /// An array is of rank 1.
    /// A matrix (e.g., 4x3 is rank 2).
    USE_RETURN size_t rank() const {
        if(_dimensions.size() == 1 && _data.size() == 1)
            return 0;
        return _dimensions.size();
    }

    /// Prints the dimensionality.
    /// E.g., for a 4 x 3 matrix tensor {4,3}
    void print_dimensionality() const {
        std::cout << _dimensions.to_string() << std::endl;
    }

    /// Interprets tensor as scalar.
    /// Fails if tensor isn't a scalar (hasn't rank 0 conditions fulfilled).
    USE_RETURN tensor_type& scalar_value() {
        if(rank() != 0)
            throw std::invalid_argument("rank must be 0");
        return this->operator[]({0});
    }

    /// Returns the maximum possible index of the linearized array.
    /// TODO: Question to myself, should be the same as _data.size() - 1?
    USE_RETURN size_t max_index() const {
        return _dimensions.multiplied_sum() - 1;
    }

    /// Resizes the vector to other dimensions.
    /// Only works if new size can be transferred into the same size of linearized array.
    void resize(vector<size_t>&& dimensions) {
        if(_dimensions.multiplied_sum() != dimensions.multiplied_sum())
            throw std::invalid_argument("can't resize tensor");
        _dimensions = std::move(_dimensions);
    }

    void set_pixel(unsigned long r, unsigned long c, vector<tensor_type> pixel) {
        if(this->rank() != 3)
            throw std::invalid_argument("rank 3 required");
        for(unsigned long i{0}; i < pixel.size(); ++i)
            operator[]({r, c, i}) = pixel[i];
    }

    // Iterators over the linearized array.
    USE_RETURN tensor_iterator begin() { return _data.begin(); }
    USE_RETURN tensor_iterator end() { return _data.end(); }

    // Const iterators over the linearized arrays.
    USE_RETURN const_tensor_iterator begin() const { return _data.begin(); }
    USE_RETURN const_tensor_iterator end() const { return _data.end(); }
};

#endif //TENSOR_H