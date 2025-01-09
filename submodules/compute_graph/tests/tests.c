#include "unity.h"
#include "compute_graph.h"

void setUp() {
    // set stuff up here
}

void tearDown() {
    // clean stuff up here
}

void scalar_compute_node_test() {
    CN_PTR node = VAR(4.5);
    TEST_ASSERT_EQUAL(value(node), 4.5);
    TEST_ASSERT_TRUE(compute_node_is_scalar(node));
    TEST_ASSERT_FALSE(compute_node_is_constant(node));
    set_value(node, 10.1);
    TEST_ASSERT_EQUAL(value(node), 10.1);
    TEST_ASSERT_EQUAL(4 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_TRUE(compute_node_is_scalar(node));
    TEST_ASSERT_FALSE(compute_node_is_constant(node));
    free_compute_node(node);
}

void constant_compute_node_test() {
    CN_PTR node = CONST(4.5);
    TEST_ASSERT_EQUAL(value(node), 4.5);
    TEST_ASSERT_FALSE(compute_node_is_scalar(node));
    TEST_ASSERT_TRUE(compute_node_is_constant(node));
    set_value(node, 10.1);
    TEST_ASSERT_EQUAL(value(node), 10.1);
    TEST_ASSERT_EQUAL(4 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_FALSE(compute_node_is_scalar(node));
    TEST_ASSERT_TRUE(compute_node_is_constant(node));
    free_compute_node(node);
}

void test_operator_compute_node() {
    CN_PTR node1 = VAR(4.5);
    CN_PTR node2 = VAR(5.5);
    CN_PTR op = SUM(node1, node2);

    TEST_ASSERT_NOT_EQUAL(node1, node2);
    TEST_ASSERT_NOT_EQUAL(value(node1), value(node2));

    TEST_ASSERT_EQUAL(*(CN_OP_TYPE*)(op->op), CN_OP_SUM);
    TEST_ASSERT_NOT_EQUAL(op->first, NULL);
    TEST_ASSERT_NOT_EQUAL(op->second, NULL);

    TEST_ASSERT_NOT_EQUAL(op->cache, NULL);

    struct CN_CACHE* cache = op->cache;
    TEST_ASSERT_EQUAL(cache->first, NULL);
    TEST_ASSERT_EQUAL(cache->second, NULL);
    TEST_ASSERT_EQUAL(cache->gradient, NULL);
}

void test_forward_run() {
    CN_PTR a = VAR(4.5);
    TEST_ASSERT_EQUAL(4.5, run_compute_graph_forward(a));

    CN_PTR b = VAR(5.5);
    TEST_ASSERT_EQUAL(5.5, run_compute_graph_forward(b));

    CN_PTR sum = SUM(a, b);
    CN_TYPE result = run_compute_graph_forward(sum);

    TEST_ASSERT_EQUAL(4.5, get_cache(1, sum));
    TEST_ASSERT_EQUAL(5.5, get_cache(2, sum));

    TEST_ASSERT_EQUAL(10., result);
}

void mat_compute_tensor_tests() {
    CT_PTR tensor = create_mat_compute_tensor(10, 15);
    TEST_ASSERT_EQUAL(150, tensor->length);

    unsigned int indices[2];
    indices[0] = 1;
    indices[1] = 2;

    TEST_ASSERT_EQUAL(17, transform_indices(tensor, indices));

    CN_PTR node = CONST(5);
    insert_into_mat_compute_tensor(tensor, node, 9, 14);
    TEST_ASSERT_EQUAL(5, value(get_mat_compute_tensor_value(tensor, 9, 14)));
    TEST_ASSERT_NULL(get_mat_compute_tensor_value(tensor, 0, 0));
}

// not needed when using generate_test_runner.rb
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(scalar_compute_node_test);
    RUN_TEST(constant_compute_node_test);
    RUN_TEST(test_operator_compute_node);
    RUN_TEST(test_forward_run);
    RUN_TEST(mat_compute_tensor_tests);
    return UNITY_END();
}