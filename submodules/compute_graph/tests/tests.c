#include "unity.h"
#include "compute_graph.h"
#include <malloc/malloc.h>

void setUp() {
    // set stuff up here
}

void tearDown() {
    // clean stuff up here
}

void scalar_compute_node_test() {
    ComputeNodeRef node = VAR(4.5);
    TEST_ASSERT_EQUAL(cnUnwrap(node), 4.5);
    TEST_ASSERT_TRUE(cnIsVariable(node));
    TEST_ASSERT_FALSE(cnIsConstant(node));
    cnSet(node, 10.1);
    TEST_ASSERT_EQUAL(cnUnwrap(node), 10.1);
    TEST_ASSERT_EQUAL(4 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_TRUE(cnIsVariable(node));
    TEST_ASSERT_FALSE(cnIsConstant(node));
    cnFree(node);
}

void constant_compute_node_test() {
    ComputeNodeRef node = CONST(4.5);
    TEST_ASSERT_EQUAL(cnUnwrap(node), 4.5);
    TEST_ASSERT_FALSE(cnIsVariable(node));
    TEST_ASSERT_TRUE(cnIsConstant(node));
    cnSet(node, 10.1);
    TEST_ASSERT_EQUAL(cnUnwrap(node), 10.1);
    TEST_ASSERT_EQUAL(4 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_FALSE(cnIsVariable(node));
    TEST_ASSERT_TRUE(cnIsConstant(node));
    cnFree(node);
}

void test_operator_compute_node() {
    ComputeNodeRef node1 = VAR(4.5);
    ComputeNodeRef node2 = VAR(5.5);
    ComputeNodeRef op = ADD(node1, node2);

    TEST_ASSERT_NOT_EQUAL(node1, node2);
    TEST_ASSERT_NOT_EQUAL(cnUnwrap(node1), cnUnwrap(node2));

    TEST_ASSERT_EQUAL(*(ComputeNodeOperatorType*)(op->op), CN_OP_SUM);
    TEST_ASSERT_NOT_EQUAL(op->first, NULL);
    TEST_ASSERT_NOT_EQUAL(op->second, NULL);

    TEST_ASSERT_NOT_EQUAL(op->cache, NULL);

    OpComputeNodeCache* cache = op->cache;
    TEST_ASSERT_EQUAL(cache->first, NULL);
    TEST_ASSERT_EQUAL(cache->second, NULL);
    TEST_ASSERT_EQUAL(cache->gradient, NULL);
}

void test_forward_run() {
    ComputeNodeRef a = VAR(4.5);
    TEST_ASSERT_EQUAL(4.5, cg_forward(a));

    ComputeNodeRef b = VAR(5.5);
    TEST_ASSERT_EQUAL(5.5, cg_forward(b));

    ComputeNodeRef sum = ADD(a, b);
    ComputeNodeValue result = cg_forward(sum);

    TEST_ASSERT_EQUAL(4.5, cnGetCache(1, sum));
    TEST_ASSERT_EQUAL(5.5, cnGetCache(2, sum));

    TEST_ASSERT_EQUAL(10., result);
}

void mat_compute_tensor_tests() {
    CT_PTR tensor = create_mat_compute_tensor(10, 15);
    TEST_ASSERT_EQUAL(150, tensor->length);

    unsigned int indices[2];
    indices[0] = 1;
    indices[1] = 2;

    TEST_ASSERT_EQUAL(17, transform_indices(tensor, indices));

    ComputeNodeRef node = CONST(5);
    insert_into_mat_compute_tensor(tensor, node, 9, 14);
    TEST_ASSERT_EQUAL(5, cnUnwrap(get_mat_compute_tensor_value(tensor, 9, 14)));
    TEST_ASSERT_NULL(get_mat_compute_tensor_value(tensor, 0, 0));
}

void scalar_compute_tensor_tests() {
    CT_PTR tensor = create_scalar_compute_tensor(CONST(5));

    TEST_ASSERT_EQUAL(1, tensor->length);
    TEST_ASSERT_EQUAL(0, tensor->rank);
    TEST_ASSERT_EQUAL(5, cnUnwrap(get_compute_tensor_value(tensor, NULL)));
}

void access_tensor_tests() {
    CT_PTR tensor = create_mat_compute_tensor(10, 15);
    TEST_ASSERT_EQUAL(150, tensor->length);
    TEST_ASSERT_EQUAL(2, tensor->rank);

    for(unsigned int i = 0; i < tensor->dimensions[1]; ++i)
        insert_into_mat_compute_tensor(tensor, VAR(i), 4, i);

    for(unsigned int i = 0; i < tensor->dimensions[1]; ++i)
        TEST_ASSERT_EQUAL(i, cnUnwrap(get_mat_compute_tensor_value(tensor, 4, i)));

    CT_PTR view = access_tensor(tensor, 4);
    TEST_ASSERT_EQUAL(15, view->length);
    TEST_ASSERT_EQUAL(1, view->rank);
    TEST_ASSERT_EQUAL(15, view->dimensions[0]);

    for(unsigned int i = 0; i < view->dimensions[0]; ++i)
        TEST_ASSERT_EQUAL(i, cnUnwrap(get_vec_compute_tensor_value(view, i)));

    free(get_vec_compute_tensor_value(view, 5));
    insert_into_vec_compute_tensor(view, CONST(14), 5);
    TEST_ASSERT_EQUAL(14, cnUnwrap(get_mat_compute_tensor_value(tensor, 4, 5)));

    CT_PTR scalar_node = access_tensor(view, 5);
    TEST_ASSERT_EQUAL(14, cnUnwrap(get_compute_tensor_value(scalar_node, NULL)));
}

void compute_tensor_free_tests() {
    CT_PTR tensor = create_mat_compute_tensor(10, 15);
    free_compute_tensor(tensor);
    TEST_ASSERT_EQUAL(0, malloc_size(tensor));
    CT_PTR scalar_tensor = create_scalar_compute_tensor(CONST(4));
    free_compute_tensor(scalar_tensor);
    TEST_ASSERT_EQUAL(0, malloc_size(scalar_tensor));
}

// not needed when using generate_test_runner.rb
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(scalar_compute_node_test);
    RUN_TEST(constant_compute_node_test);
    RUN_TEST(test_operator_compute_node);
    RUN_TEST(test_forward_run);
    RUN_TEST(mat_compute_tensor_tests);
    RUN_TEST(scalar_compute_tensor_tests);
    RUN_TEST(access_tensor_tests);
    RUN_TEST(compute_tensor_free_tests);
    return UNITY_END();
}