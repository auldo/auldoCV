#ifndef VECTOR_H
#define VECTOR_H

#define USE_RETURN [[nodiscard]]

#include <memory>
#include <string>
#include "arithmetic.h"

template <typename tensor_type> requires arithmetic<tensor_type>
class vector {

    // Iterator types
    using array_iterator = tensor_type*;
    using const_array_iterator = const tensor_type*;

    /// The vector's current size (i.e. the number of elements stored in that vector)
    size_t _size;

    /// The actual elements stored in that vector
    std::unique_ptr<tensor_type[]> _data{std::make_unique<tensor_type[]>(0)};
public:

    /// Creates an empty vector.
    vector() : _size(0) {}

    /// Creates a vector containing the elements in the initializer list.
    vector(std::initializer_list<tensor_type> init) : _size(init.size()), _data(std::make_unique<tensor_type[]>(init.size())) {
        for(auto i{0}; i < init.size(); ++i)
            _data[i] = *(init.begin() + i);
    }

    /// Creates a vector of a certain size, without initializing values.
    /// Note, that anything could be stored in those indices.
    explicit vector(const size_t size) : _size(size), _data(std::make_unique<tensor_type[]>(size)) {}

    /// Moves the vector to the assigned variable, leaving the moved vector empty.
    vector &operator=(vector &&other)  noexcept {
        this->reset_size(other._size);
        this->_data = std::move(other._data);
        other.reset_size(0);
        return *this;
    }

    /**
    * Vectors need to be moved when being returned from functions.
    */
    vector(vector &&other) noexcept: _size(other._size), _data(std::move(other._data)) {}

    // Removed copy constructor and assignment operator.
    vector(const vector &other) = delete;
    vector &operator=(const vector &other) = delete;

    // Iterators
    USE_RETURN array_iterator begin() { return _data.get(); }
    USE_RETURN array_iterator end() { return _data.get() + _size; }

    // Const iterators
    USE_RETURN const_array_iterator begin() const { return _data.get(); }
    USE_RETURN const_array_iterator end() const { return _data.get() + _size; }

    /// Sets size to a certain size.
    /// Doesn't care about what's in the vector, so data may be "lost".
    void reset_size(size_t size) {
        this->_size = size;
        this->_data = std::make_unique<tensor_type[]>(size);
    }

    /// Accesses element at certain index, may throw out of range.
    tensor_type& operator[](size_t idx) {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    /// Const-access to element at certain index, may throw out of range.
    const tensor_type& operator[](size_t idx) const {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    /// Required for tensor functionality.
    /// Multiplies the elements in the vector.
    /// A linearized tensor of dimensionality 3 x 4 x 4 needs capacity to store 3*4*4 elements.
    USE_RETURN tensor_type multiplied_sum() const {
        tensor_type sum{this->_size == 0 ? static_cast<tensor_type>(0) : static_cast<tensor_type>(1)};
        for(auto& elem : *this)
            sum *= elem;
        return sum;
    }

    /// Works similar as multiplied_sum but only takes the last n indices into account.
    /// Required for tensor index transformation.
    USE_RETURN tensor_type multiplied_sum_last_n(const size_t n) const {
        tensor_type sum{this->_size == 0 ? static_cast<tensor_type>(0) : static_cast<tensor_type>(1)};
        for(auto i{0}; i < n; ++i) {
            sum *= this->operator[](this->_size - 1 - i);
        }
        return sum;
    }

    /// Returns the size of the array.
    USE_RETURN size_t size() const { return _size; }

    /// Appends array elements and returns the built string.
    USE_RETURN std::string to_string() const {
        std::string result{"["};
        for(const auto& elem : *this) {
            result += std::to_string(elem) + ",";
        }
        if(_size > 0)
            result.pop_back();
        return result + "]";
    }
};

#endif