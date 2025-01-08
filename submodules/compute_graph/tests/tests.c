#include "unity.h"
#include "compute_node.h"

void setUp() {
    // set stuff up here
}

void tearDown() {
    // clean stuff up here
}

void scalar_compute_node_test() {
    CN_PTR node = create_scalar_compute_node(4.5);
    TEST_ASSERT_EQUAL(value(node), 4.5);
    TEST_ASSERT_TRUE(compute_node_is_scalar(node));
    TEST_ASSERT_FALSE(compute_node_is_constant(node));
    set_value(node, 10.1);
    TEST_ASSERT_EQUAL(value(node), 10.1);
    TEST_ASSERT_EQUAL(3 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_TRUE(compute_node_is_scalar(node));
    TEST_ASSERT_FALSE(compute_node_is_constant(node));
    free_compute_node(node);
}

void constant_compute_node_test() {
    CN_PTR node = create_constant_compute_node(4.5);
    TEST_ASSERT_EQUAL(value(node), 4.5);
    TEST_ASSERT_FALSE(compute_node_is_scalar(node));
    TEST_ASSERT_TRUE(compute_node_is_constant(node));
    set_value(node, 10.1);
    TEST_ASSERT_EQUAL(value(node), 10.1);
    TEST_ASSERT_EQUAL(3 * sizeof(NULL), sizeof(*node));
    TEST_ASSERT_FALSE(compute_node_is_scalar(node));
    TEST_ASSERT_TRUE(compute_node_is_constant(node));
    free_compute_node(node);
}

// not needed when using generate_test_runner.rb
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(scalar_compute_node_test);
    RUN_TEST(constant_compute_node_test);
    return UNITY_END();
}