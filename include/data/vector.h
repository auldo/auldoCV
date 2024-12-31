#pragma once

#include "constants.h"

template <typename VECTOR_TYPE>
class Vector {

    // Iterator types
    using array_iterator = VECTOR_TYPE*;
    using const_array_iterator = const VECTOR_TYPE*;

    /// The vector's current size (i.e. the number of elements stored in that vector)
    INDEX_NBR _size;

    /// The actual elements stored in that vector
    std::unique_ptr<VECTOR_TYPE[]> _data{std::unique_ptr<VECTOR_TYPE[]>(0)};
public:

    /// Creates an empty vector.
    Vector() : _size(0) {}

    /// Creates a vector containing the elements in the initializer list.
    Vector(std::initializer_list<VECTOR_TYPE> init) : _size(init.size()), _data(std::make_unique<VECTOR_TYPE[]>(init.size())) {
        for(INDEX_NBR i{0}; i < init.size(); ++i)
            _data[i] = *(init.begin() + i);
    }

    /// Creates a vector of a certain size, without initializing values.
    /// Note, that anything could be stored in those indices.
    explicit Vector(const INDEX_NBR size) : _size(size), _data(std::make_unique<VECTOR_TYPE[]>(size)) {}

    Vector(const INDEX_NBR size, VECTOR_TYPE fillValue): Vector(size) {
        fill(fillValue);
    }

    /// Moves the vector to the assigned variable, leaving the moved vector empty.
    Vector &operator=(Vector &&other)  noexcept {
        this->resize(other._size);
        this->_data = std::move(other._data);
        other.resize(0);
        return *this;
    }

    /**
    * Vectors need to be moved when being returned from functions.
    */
    Vector(Vector &&other) noexcept: _size(other._size), _data(std::move(other._data)) {}

    // Removed copy constructor and assignment operator.
    Vector(const Vector &other) = delete;
    Vector &operator=(const Vector &other) = delete;

    // Iterators
    USE_RETURN array_iterator begin() { return _data.get(); }
    USE_RETURN array_iterator end() { return _data.get() + _size; }

    // Const iterators
    USE_RETURN const_array_iterator begin() const { return _data.get(); }
    USE_RETURN const_array_iterator end() const { return _data.get() + _size; }

    /// Sets size to a certain size.
    /// Doesn't care about what's in the vector, so data may be "lost".
    void resize(INDEX_NBR size) {
        this->_size = size;
        this->_data = std::make_unique<VECTOR_TYPE[]>(size);
    }

    /// Accesses element at certain index, may throw out of range.
    VECTOR_TYPE& at(INDEX_NBR idx) {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    /// Const-access to element at certain index, may throw out of range.
    const VECTOR_TYPE& at(INDEX_NBR idx) const {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    /// Required for tensor functionality.
    /// Multiplies the elements in the vector.
    /// A linearized tensor of dimensionality 3 x 4 x 4 needs capacity to store 3*4*4 elements.
    USE_RETURN VECTOR_TYPE multiplied_sum() const {
        VECTOR_TYPE sum{this->_size == 0 ? static_cast<VECTOR_TYPE>(0) : static_cast<VECTOR_TYPE>(1)};
        for(auto& elem : *this)
            sum *= elem;
        return sum;
    }

    /// Works similar as multiplied_sum but only takes the last n indices into account.
    /// Required for tensor index transformation.
    USE_RETURN VECTOR_TYPE multiplied_sum_last_n(const INDEX_NBR n) const {
        VECTOR_TYPE sum{this->_size == 0 ? static_cast<VECTOR_TYPE>(0) : static_cast<VECTOR_TYPE>(1)};
        for(auto i{0}; i < n; ++i) {
            sum *= this->at(this->_size - 1 - i);
        }
        return sum;
    }

    /// Returns the size of the array.
    USE_RETURN INDEX_NBR size() const { return _size; }

    void fill(const VECTOR_TYPE value) {
        for(auto& elem : *this)
            elem = value;
    }
};