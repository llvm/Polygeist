#define _POSIX_C_SOURCE 200809L
#define main polygeist_llama_fixture_original_main
#ifndef LLAMA_HARNESS_NATIVE
#define kernel_llama2_extended_forward kernel_llama2_extended_forward_native
#endif
#include "../../third_party/cnn-extracted/llama2_extended_forward_bench.c"
#ifndef LLAMA_HARNESS_NATIVE
#undef kernel_llama2_extended_forward
#endif
#undef main

#ifndef LLAMA_HARNESS_NATIVE
/* Implemented by the compiler-generated ABI wrapper linked with this harness. */
extern void kernel_llama2_extended_forward();
#endif

#include <float.h>
#include <stdlib.h>
#include <time.h>

#ifndef WARMUP
#define WARMUP 5
#endif

#ifndef MEASURE
#define MEASURE 5
#endif

#ifndef DUMP_ALL
#define DUMP_ALL 0
#endif

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int compare_double(const void *lhs, const void *rhs) {
  const double a = *(const double *)lhs;
  const double b = *(const double *)rhs;
  return (a > b) - (a < b);
}

static void run_kernel_once(void) {
  const int token = 7;
  const int pos = SEQ_LEN / 2;
  kernel_llama2_extended_forward(
      token, pos, tok_embeddings, rms_att_weight, wq_even, wq_odd, wk_even,
      wk_odd, wv, wo, rms_ffn_weight, w_gate, w_up, w_down,
      rms_final_weight, lm_head, cos_table, sin_table, k_cache_even,
      k_cache_odd, v_cache, x, att_normed, v, q_even, q_odd, k_even, k_odd,
      q_even_rot, q_odd_rot, k_read_even, k_read_odd, v_read, scores,
      masked_scores, probs, att_out, proj_out, resid_att, ffn_normed, gate,
      up, ffn_hidden, ffn_out, resid_ffn, final_normed, logits);
}

int main(void) {
  double samples[MEASURE];
  init_array();
  for (int i = 0; i < WARMUP; ++i)
    run_kernel_once();
  for (int i = 0; i < MEASURE; ++i) {
    const double start = now_ms();
    run_kernel_once();
    samples[i] = now_ms() - start;
  }

  double sum_ms = 0.0;
  for (int i = 0; i < MEASURE; ++i)
    sum_ms += samples[i];
  qsort(samples, MEASURE, sizeof(double), compare_double);
  const double median_ms = MEASURE % 2
      ? samples[MEASURE / 2]
      : 0.5 * (samples[MEASURE / 2 - 1] + samples[MEASURE / 2]);

  double checksum = 0.0;
  double sumsq = 0.0;
  double maxabs = 0.0;
  for (int i = 0; i < VOCAB; ++i) {
    const double value = (double)logits[i];
    const double magnitude = fabs(value);
    checksum += value;
    sumsq += value * value;
    if (magnitude > maxabs)
      maxabs = magnitude;
  }

  printf("implementation,model_dim,ffn_dim,vocab,seq_len,heads,token,pos,"
         "warmup,iters,mean_ms,median_ms,min_ms,max_ms,checksum,sumsq,maxabs\n");
  printf("llama_extended_c,%d,%d,%d,%d,%d,7,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,"
         "%.17g,%.17g,%.17g\n",
         MODEL_DIM, FFN_DIM, VOCAB, SEQ_LEN, NUM_HEADS, SEQ_LEN / 2,
         WARMUP, MEASURE, sum_ms / (double)MEASURE, median_ms,
         samples[0], samples[MEASURE - 1], checksum, sumsq, maxabs);
  for (int i = 0; i < MEASURE; ++i)
    printf("TIMING_SAMPLE,%d,%.9f\n", i, samples[i]);
  for (int i = 0; i < 8 && i < VOCAB; ++i)
    printf("SAMPLE,%d,%.9g\n", i, (double)logits[i]);
  if (DUMP_ALL)
    for (int i = 0; i < VOCAB; ++i)
      printf("LOGIT,%d,%.9g\n", i, (double)logits[i]);
  return 0;
}
