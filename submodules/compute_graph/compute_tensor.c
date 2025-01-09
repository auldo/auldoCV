#include "compute_tensor.h"

CT_PTR create_compute_tensor(unsigned int rank, unsigned int* indices) {
    if(rank == 0) {
        THROW("rank 0 tensors not supported, yet")
    }

    unsigned int* dimensions = malloc(rank * sizeof(unsigned int));
    memcpy(dimensions, indices, rank * sizeof(unsigned int));
    unsigned int length = 1;
    for (unsigned int i = 0; i < rank; i++)
        length *= dimensions[i];

    CN_PTR* nodes = malloc(length * sizeof(CN_PTR));
    for(unsigned int i = 0; i < length; ++i) {
        nodes[i] = NULL;
    }

    CT_PTR tensor = malloc(sizeof(struct CT));
    tensor->rank = rank;
    tensor->dimensions = dimensions;
    tensor->data = nodes;
    tensor->length = length;
    return tensor;
}

CT_PTR create_mat_compute_tensor(unsigned int rows, unsigned int cols) {
    unsigned int dimensions[2];
    dimensions[0] = rows;
    dimensions[1] = cols;
    return create_compute_tensor(2, dimensions);
}

unsigned int transform_indices(const CT_PTR tensor, unsigned int* indices) {
    if(tensor->rank == 0)
        THROW("can't transform indices for tensor of rank 0");
    if(tensor->rank == 1)
        return indices[0];
    unsigned int result = 0;
    for(unsigned int i = 0; i < tensor->rank; ++i) {
        unsigned int idx = tensor->rank - i - 1;
        unsigned int sum = 1;
        for(unsigned int i2 = idx + 1; i2 < tensor->rank; ++i2) {
            sum *= tensor->dimensions[i2];
        }
        result += (indices[idx] * sum);
    }
    return result;
}

void insert_into_compute_tensor(CT_PTR tensor, CN_PTR node, unsigned int* indices) {
    unsigned int index = transform_indices(tensor, indices);
    if(index >= tensor->length)
        THROW("out of bounds");
    tensor->data[index] = node;
}

void insert_into_mat_compute_tensor(CT_PTR tensor, CN_PTR node, unsigned int rows, unsigned int cols) {
    unsigned int indices[2];
    indices[0] = rows;
    indices[1] = cols;
    return insert_into_compute_tensor(tensor, node, indices);
}

CN_PTR get_compute_tensor_value(const CT_PTR tensor, unsigned int* indices) {
    unsigned int index = transform_indices(tensor, indices);
    if(index >= tensor->length)
        THROW("out of bounds");
    return tensor->data[index];
}

CN_PTR get_mat_compute_tensor_value(const CT_PTR tensor, unsigned int rows, unsigned int cols) {
    unsigned int indices[2];
    indices[0] = rows;
    indices[1] = cols;
    return get_compute_tensor_value(tensor, indices);
}