#include "unity.h"
#include "compute_graph.h"

void setUp() {
    // set stuff up here
}

void tearDown() {
    // clean stuff up here
}

void scalar_compute_node_test() {
    CN_PTR node = create_scalar_compute_node(4.5);
    TEST_ASSERT_TRUE(compute_node_is_scalar(node));
    TEST_ASSERT_EQUAL(scalar(node), 4.5);
}

// not needed when using generate_test_runner.rb
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(scalar_compute_node_test);
    return UNITY_END();
}