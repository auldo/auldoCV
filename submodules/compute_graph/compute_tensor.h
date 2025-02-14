#ifndef AULDO_CV_COMPUTE_TENSOR_H
#define AULDO_CV_COMPUTE_TENSOR_H

#include "compute_node.h"

#define CT compute_tensor
#define CT_PTR struct compute_tensor*

struct CT {

    /// 0 for scalar = single compute node
    /// 1 for list of compute nodes
    /// 2 for matrix of compute nodes, 3 ..., 4 ..., etc.
    unsigned int rank;

    /// NULL if rank = 0.
    unsigned int* dimensions;

    unsigned int length;

    ComputeNodeRef* data;
};

void free_compute_tensor(CT_PTR tensor);

unsigned int transform_indices(const CT_PTR tensor, unsigned int* indices);

CT_PTR create_compute_tensor(unsigned int rank, unsigned int* indices);
CT_PTR create_scalar_compute_tensor(ComputeNodeRef node);
CT_PTR create_mat_compute_tensor(unsigned int rows, unsigned int cols);

CT_PTR access_tensor(const CT_PTR tensor, int index);

void insert_into_compute_tensor(CT_PTR tensor, ComputeNodeRef node, unsigned int* indices);
void insert_into_mat_compute_tensor(CT_PTR tensor, ComputeNodeRef node, unsigned int rows, unsigned int cols);
void insert_into_vec_compute_tensor(CT_PTR tensor, ComputeNodeRef node, unsigned int index);

ComputeNodeRef get_compute_tensor_value(const CT_PTR tensor, unsigned int* indices);
ComputeNodeRef get_mat_compute_tensor_value(const CT_PTR tensor, unsigned int rows, unsigned int cols);
ComputeNodeRef get_vec_compute_tensor_value(const CT_PTR tensor, unsigned int index);

#endif //AULDO_CV_COMPUTE_TENSOR_H