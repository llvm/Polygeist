/* Thin state adapter between the original NPB IS driver and raised rank. */
#include "npbparams.h"

#define TOTAL_KEYS (1 << 16)
#define MAX_KEY (1 << 11)
#define NUM_BUCKETS (1 << 9)
#define MAX_ITERATIONS 10
#define TEST_ARRAY_SIZE 5

extern int *key_buff_ptr_global;
extern int passed_verification;
extern int key_array[TOTAL_KEYS];
extern int key_buff1[MAX_KEY];
extern int key_buff2[TOTAL_KEYS];
extern int partial_verify_vals[TEST_ARRAY_SIZE];
extern int bucket_size[NUM_BUCKETS];
extern int bucket_ptrs[NUM_BUCKETS];
extern int test_index_array[TEST_ARRAY_SIZE];
extern int test_rank_array[TEST_ARRAY_SIZE];

extern void npb_is_rank_core(
    int iteration, int *key_array, int *partial_verify_vals,
    const int *test_index_array, int *bucket_size, int *bucket_ptrs,
    int *key_buff2, int *key_buff1, const int *test_rank_array,
    int *passed_verification);

void rank(int iteration) {
  npb_is_rank_core(iteration, key_array, partial_verify_vals,
                   test_index_array, bucket_size, bucket_ptrs, key_buff2,
                   key_buff1, test_rank_array, &passed_verification);
  if (iteration == MAX_ITERATIONS)
    key_buff_ptr_global = key_buff1;
}
