#define _POSIX_C_SOURCE 200809L
#define main polygeist_llama_fixture_original_main
#define kernel_llama2_extended_forward_session \
  kernel_llama2_extended_forward_session_native
#include "../../third_party/cnn-extracted/llama2_extended_forward_bench.c"
#undef kernel_llama2_extended_forward_session
#undef main

/* Implemented by the generated ABI wrapper linked with this harness. */
extern void kernel_llama2_extended_forward_session();

#include <time.h>

#ifndef SESSION_REPETITIONS
#define SESSION_REPETITIONS 100
#endif

#ifndef DUMP_ALL
#define DUMP_ALL 0
#endif

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

int main(void) {
  const int token = 7;
  const int pos = SEQ_LEN / 2;
  init_array();

  const double start = now_ms();
  kernel_llama2_extended_forward_session(
      SESSION_REPETITIONS, token, pos, tok_embeddings, rms_att_weight, wq_even,
      wq_odd, wk_even, wk_odd, wv, wo, rms_ffn_weight, w_gate, w_up, w_down,
      rms_final_weight, lm_head, cos_table, sin_table, k_cache_even,
      k_cache_odd, v_cache, x, att_normed, v, q_even, q_odd, k_even, k_odd,
      q_even_rot, q_odd_rot, k_read_even, k_read_odd, v_read, scores,
      masked_scores, probs, att_out, proj_out, resid_att, ffn_normed, gate, up,
      ffn_hidden, ffn_out, resid_ffn, final_normed, logits);
  const double total_ms = now_ms() - start;

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
         "session_repetitions,total_ms,amortized_ms,checksum,sumsq,maxabs\n");
  printf("llama_extended_resident_session,%d,%d,%d,%d,%d,%d,%d,%d,"
         "%.9f,%.9f,%.17g,%.17g,%.17g\n",
         MODEL_DIM, FFN_DIM, VOCAB, SEQ_LEN, NUM_HEADS, token, pos,
         SESSION_REPETITIONS, total_ms,
         total_ms / (double)SESSION_REPETITIONS, checksum, sumsq, maxabs);
  for (int i = 0; i < 8 && i < VOCAB; ++i)
    printf("SAMPLE,%d,%.9g\n", i, (double)logits[i]);
  if (DUMP_ALL)
    for (int i = 0; i < VOCAB; ++i)
      printf("LOGIT,%d,%.9g\n", i, (double)logits[i]);
  return 0;
}
