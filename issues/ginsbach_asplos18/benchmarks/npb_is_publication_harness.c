#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void npb_is_rank_core(int iteration, int *key_array,
                      int *partial_verify_vals,
                      const int *test_index_array, int *bucket_size,
                      int *bucket_ptrs, int *key_buff2, int *key_buff1,
                      const int *test_rank_array, int *passed_verification);
void npb_is_reference(int iteration, int *key_array, int *partial_verify_vals,
                      const int *test_index_array, int *bucket_size,
                      int *bucket_ptrs, int *key_buff2, int *key_buff1,
                      const int *test_rank_array, int *passed_verification);

enum {
  kWarmups = 5,
  kSamples = 5,
  kNumKeys = 65536,
  kMaxKey = 2048,
  kBuckets = 512,
  kIteration = 1
};

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

static int equal_ints(const int *left, const int *right, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (left[i] != right[i])
      return 0;
  return 1;
}

int main(void) {
  int *initial_keys = malloc((size_t)kNumKeys * sizeof(int));
  int *keys = malloc((size_t)kNumKeys * sizeof(int));
  int *expected_keys = malloc((size_t)kNumKeys * sizeof(int));
  int *bucket_size = malloc((size_t)kBuckets * sizeof(int));
  int *bucket_ptrs = malloc((size_t)kBuckets * sizeof(int));
  int *key_buff2 = malloc((size_t)kNumKeys * sizeof(int));
  int *key_buff1 = malloc((size_t)kMaxKey * sizeof(int));
  int *expected_bucket_size = malloc((size_t)kBuckets * sizeof(int));
  int *expected_bucket_ptrs = malloc((size_t)kBuckets * sizeof(int));
  int *expected_buff2 = malloc((size_t)kNumKeys * sizeof(int));
  int *expected_buff1 = malloc((size_t)kMaxKey * sizeof(int));
  if (!initial_keys || !keys || !expected_keys || !bucket_size ||
      !bucket_ptrs || !key_buff2 || !key_buff1 || !expected_bucket_size ||
      !expected_bucket_ptrs || !expected_buff2 || !expected_buff1)
    return 100;

  for (int i = 0; i < kNumKeys; ++i)
    initial_keys[i] = (i * 37 + i / 17) & (kMaxKey - 1);
  const int test_index[5] = {101, 1009, 8191, 32771, 60013};
  int test_rank[5] = {0, 0, 0, 0, 0};
  int histogram[kMaxKey];
  memset(histogram, 0, sizeof(histogram));
  memcpy(expected_keys, initial_keys, (size_t)kNumKeys * sizeof(int));
  expected_keys[kIteration] = kIteration;
  expected_keys[kIteration + 10] = kMaxKey - kIteration;
  for (int i = 0; i < kNumKeys; ++i)
    histogram[expected_keys[i]]++;
  for (int i = 1; i < kMaxKey; ++i)
    histogram[i] += histogram[i - 1];
  for (int i = 0; i < 5; ++i) {
    int key = expected_keys[test_index[i]];
    int rank = key > 0 ? histogram[key - 1] : 0;
    test_rank[i] = i <= 2 ? rank - kIteration : rank + kIteration;
  }

  int partial[5], passed = 0;
  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
    memcpy(keys, initial_keys, (size_t)kNumKeys * sizeof(int));
    passed = 0;
    double begin = wall_ms();
    npb_is_rank_core(kIteration, keys, partial, test_index, bucket_size,
                     bucket_ptrs, key_buff2, key_buff1, test_rank, &passed);
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }

  int expected_partial[5], expected_passed = 0;
  memcpy(expected_keys, initial_keys, (size_t)kNumKeys * sizeof(int));
  npb_is_reference(kIteration, expected_keys, expected_partial, test_index,
                   expected_bucket_size, expected_bucket_ptrs, expected_buff2,
                   expected_buff1, test_rank, &expected_passed);
  int ok = passed == expected_passed &&
           equal_ints(keys, expected_keys, kNumKeys) &&
           equal_ints(partial, expected_partial, 5) &&
           equal_ints(bucket_size, expected_bucket_size, kBuckets) &&
           equal_ints(bucket_ptrs, expected_bucket_ptrs, kBuckets) &&
           equal_ints(key_buff2, expected_buff2, kNumKeys) &&
           equal_ints(key_buff1, expected_buff1, kMaxKey);
  printf("BENCH_CORRECTNESS complete_integer_outputs=%s passed=%d expected=%d\n",
         ok ? "true" : "false", passed, expected_passed);
  printf("BENCH_RESULT workload=npb_is_rank class=S status=%s\n",
         ok ? "PASS" : "FAIL");

  free(initial_keys);
  free(keys);
  free(expected_keys);
  free(bucket_size);
  free(bucket_ptrs);
  free(key_buff2);
  free(key_buff1);
  free(expected_bucket_size);
  free(expected_bucket_ptrs);
  free(expected_buff2);
  free(expected_buff1);
  return ok ? 0 : 1;
}
