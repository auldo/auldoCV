#include "compute_tensor.h"

void free_compute_tensor(CT_PTR tensor) {
    free(tensor->dimensions);
    for(unsigned int i = 0; i < tensor->length; ++i)
        free_compute_node(tensor->data[i]);
    free(tensor->data);
    free(tensor);
}

CT_PTR create_scalar_compute_tensor(CN_PTR node) {
    CN_PTR* nodes = malloc(sizeof(CN_PTR));
    nodes[0] = node;

    CT_PTR tensor = malloc(sizeof(struct CT));
    tensor->rank = 0;
    tensor->dimensions = NULL;
    tensor->data = nodes;
    tensor->length = 1;
    return tensor;
}

CT_PTR create_compute_tensor(unsigned int rank, unsigned int* indices) {
    if(rank == 0)
        return create_scalar_compute_tensor(NULL);
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

CT_PTR access_tensor(const CT_PTR tensor, int index) {
    if(tensor->rank == 0)
        THROW("expected tensor of min rank 1");
    if(tensor->rank == 1)
        return create_scalar_compute_tensor(tensor->data[index]);
    if(index >= tensor->dimensions[0])
        THROW("out of range");
    unsigned int rank = tensor->rank - 1;
    unsigned int length = tensor->length / tensor->dimensions[0];
    unsigned int* dimensions = malloc(rank * sizeof(unsigned int));
    for(unsigned int i = 0; i < rank; ++i)
        dimensions[i] = tensor->dimensions[i + 1];

    unsigned int* indices = malloc(tensor->rank * sizeof(int));
    indices[0] = index;
    for(unsigned int i = 1; i < tensor->rank; ++i)
        indices[i] = 0;
    unsigned int start = transform_indices(tensor, indices);
    free(indices);

    CT_PTR accessed = malloc(sizeof(struct CT));
    accessed->dimensions = dimensions;
    accessed->length = length;
    accessed->rank = rank;
    accessed->data = &tensor->data[start];
    return accessed;
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

void insert_into_vec_compute_tensor(CT_PTR tensor, CN_PTR node, unsigned int index) {
    unsigned int indices[1];
    indices[0] = index;
    return insert_into_compute_tensor(tensor, node, indices);
}

CN_PTR get_compute_tensor_value(const CT_PTR tensor, unsigned int* indices) {
    if(indices == NULL && tensor->rank == 0) {
        return tensor->data[0];
    }
    if(indices == NULL)
        THROW("indices must be given unless tensor is of rank 0")
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

CN_PTR get_vec_compute_tensor_value(const CT_PTR tensor, unsigned int index) {
    unsigned int indices[1];
    indices[0] = index;
    return get_compute_tensor_value(tensor, indices);
}