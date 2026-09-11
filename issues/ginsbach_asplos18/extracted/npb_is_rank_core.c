#include <stdio.h>

// Source-faithful Class-S computational core of NPB3.3-SER-C IS/rank.
// File-static arrays are explicit arguments so the original driver can retain
// ownership of benchmark state while this routine is raised independently.
// polygeist-arg-extents npb_is_rank_core: key_array=65536, partial_verify_vals=5, test_index_array=5, bucket_size=512, bucket_ptrs=512, key_buff2=65536, key_buff1=2048, test_rank_array=5, passed_verification=1
void npb_is_rank_core(int iteration, int *key_array,
                      int *partial_verify_vals,
                      const int *test_index_array, int *bucket_size,
                      int *bucket_ptrs, int *key_buff2, int *key_buff1,
                      const int *test_rank_array,
                      int *passed_verification) {
  const int num_keys = 65536;
  const int max_key = 2048;
  const int num_buckets = 512;
  const int max_iterations = 10;
  const int shift = 2;
  int i;

  key_array[iteration] = iteration;
  key_array[iteration + max_iterations] = max_key - iteration;

  for (i = 0; i < 5; ++i)
    partial_verify_vals[i] = key_array[test_index_array[i]];

  for (i = 0; i < num_buckets; ++i)
    bucket_size[i] = 0;
  for (i = 0; i < num_keys; ++i)
    bucket_size[key_array[i] >> shift]++;

  bucket_ptrs[0] = 0;
  for (i = 1; i < num_buckets; ++i)
    bucket_ptrs[i] = bucket_ptrs[i - 1] + bucket_size[i - 1];

  for (i = 0; i < num_keys; ++i) {
    int key = key_array[i];
    key_buff2[bucket_ptrs[key >> shift]++] = key;
  }

  for (i = 0; i < max_key; ++i)
    key_buff1[i] = 0;
  for (i = 0; i < num_keys; ++i)
    key_buff1[key_buff2[i]]++;
  for (i = 0; i < max_key - 1; ++i)
    key_buff1[i + 1] += key_buff1[i];

  for (i = 0; i < 5; ++i) {
    int key = partial_verify_vals[i];
    if (0 < key && key <= num_keys - 1) {
      int key_rank = key_buff1[key - 1];
      int expected = i <= 2 ? test_rank_array[i] + iteration
                            : test_rank_array[i] - iteration;
      if (key_rank != expected)
        printf("Failed partial verification: iteration %d, test key %d\n",
               iteration, i);
      else
        (*passed_verification)++;
    }
  }
}
