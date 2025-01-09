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

    CN_PTR* data;
};

unsigned int transform_indices(CT_PTR tensor, unsigned int* indices);

CT_PTR create_compute_tensor(unsigned int rank, unsigned int* indices);
CT_PTR create_mat_compute_tensor(int rows, int cols);

//void insert_into_compute_tensor(CT_PTR tensor, CN_PTR node, unsigned int* indices);
//void insert_into_mat_ct(int r, int c);

#endif //AULDO_CV_COMPUTE_TENSOR_H