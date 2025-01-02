#pragma once

#include "data/vector.h"

class Linearized {
protected:
    Linearized() = default;
public:
    virtual ~Linearized() = default;

    explicit Linearized(Vector<INDEX_NBR> dim): _dimensions(dim){};

    /// The dimensionality of the tensor.
    /// E.g., a matrix with 3 rows and 4 columns would have the value {3, 4}.
    Vector<INDEX_NBR> _dimensions{0};

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

    USE_RETURN INDEX_NBR shapeSize(unsigned index) const {
        return _dimensions.at(index);
    }
};