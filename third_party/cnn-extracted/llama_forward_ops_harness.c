/* llama_forward_ops_harness.c -- timing harness for llama_forward_ops.c.
 *
 * This file intentionally only declares the kernels. The build driver links
 * these calls against the raised wrapper, so compiling the harness separately
 * prevents the C compiler from inlining or reasoning through the original
 * kernel body.
 */

#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef MODEL_DIM
#define MODEL_DIM 64
#endif

#ifndef FFN_DIM
#define FFN_DIM 128
#endif

#ifndef VOCAB
#define VOCAB 256
#endif

#ifndef SEQ_LEN
#define SEQ_LEN 32
#endif

#ifndef NUM_HEADS
#define NUM_HEADS 4
#endif

#ifndef HEAD_DIM
#define HEAD_DIM (MODEL_DIM / NUM_HEADS)
#endif

#ifndef HALF_HEAD_DIM
#define HALF_HEAD_DIM (HEAD_DIM / 2)
#endif

#ifndef LLAMA_OP
#error "Define LLAMA_OP to select the operation to time"
#endif

#ifndef REPEAT
#define REPEAT 50
#endif

void kernel_llama_token_embedding(int token,
                                  DATA_TYPE embedding[VOCAB][MODEL_DIM],
                                  DATA_TYPE out[MODEL_DIM]);
void kernel_llama_attention_rmsnorm(DATA_TYPE out[MODEL_DIM],
                                    DATA_TYPE x[MODEL_DIM],
                                    DATA_TYPE weight[MODEL_DIM]);
void kernel_llama_qkv_projection(DATA_TYPE x[MODEL_DIM],
                                 DATA_TYPE wq[MODEL_DIM][MODEL_DIM],
                                 DATA_TYPE wk[MODEL_DIM][MODEL_DIM],
                                 DATA_TYPE wv[MODEL_DIM][MODEL_DIM],
                                 DATA_TYPE q[MODEL_DIM],
                                 DATA_TYPE k[MODEL_DIM],
                                 DATA_TYPE v[MODEL_DIM]);
void kernel_llama_rope_split(int pos,
                             DATA_TYPE q_even[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE q_odd[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE k_even[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE k_odd[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE cos_table[SEQ_LEN][HALF_HEAD_DIM],
                             DATA_TYPE sin_table[SEQ_LEN][HALF_HEAD_DIM],
                             DATA_TYPE q_even_out[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE q_odd_out[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE k_even_out[NUM_HEADS][HALF_HEAD_DIM],
                             DATA_TYPE k_odd_out[NUM_HEADS][HALF_HEAD_DIM]);
void kernel_llama_kv_cache_rw(int pos, DATA_TYPE k[MODEL_DIM],
                              DATA_TYPE v[MODEL_DIM],
                              DATA_TYPE k_cache[SEQ_LEN][MODEL_DIM],
                              DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM],
                              DATA_TYPE k_read[SEQ_LEN][MODEL_DIM],
                              DATA_TYPE v_read[SEQ_LEN][MODEL_DIM]);
void kernel_llama_attention_scores(DATA_TYPE q[MODEL_DIM],
                                   DATA_TYPE k_cache[SEQ_LEN][MODEL_DIM],
                                   DATA_TYPE scores[SEQ_LEN]);
void kernel_llama_attention_mask_select(int pos, DATA_TYPE scores[SEQ_LEN],
                                        DATA_TYPE masked[SEQ_LEN]);
void kernel_llama_attention_softmax(DATA_TYPE out[SEQ_LEN],
                                    DATA_TYPE scores[SEQ_LEN]);
void kernel_llama_attention_output(DATA_TYPE probs[SEQ_LEN],
                                   DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM],
                                   DATA_TYPE out[MODEL_DIM]);
void kernel_llama_output_projection(DATA_TYPE x[MODEL_DIM],
                                    DATA_TYPE w[MODEL_DIM][MODEL_DIM],
                                    DATA_TYPE out[MODEL_DIM]);
void kernel_llama_residual_add(DATA_TYPE out[MODEL_DIM],
                               DATA_TYPE x[MODEL_DIM],
                               DATA_TYPE residual[MODEL_DIM]);
void kernel_llama_ffn_rmsnorm(DATA_TYPE out[MODEL_DIM],
                              DATA_TYPE x[MODEL_DIM],
                              DATA_TYPE weight[MODEL_DIM]);
void kernel_llama_gate_up_projection(DATA_TYPE x[MODEL_DIM],
                                     DATA_TYPE w_gate[FFN_DIM][MODEL_DIM],
                                     DATA_TYPE w_up[FFN_DIM][MODEL_DIM],
                                     DATA_TYPE gate[FFN_DIM],
                                     DATA_TYPE up[FFN_DIM]);
void kernel_llama_swiglu(DATA_TYPE gate[FFN_DIM], DATA_TYPE up[FFN_DIM],
                         DATA_TYPE out[FFN_DIM]);
void kernel_llama_down_projection(DATA_TYPE hidden[FFN_DIM],
                                  DATA_TYPE w[MODEL_DIM][FFN_DIM],
                                  DATA_TYPE out[MODEL_DIM]);
void kernel_llama_final_rmsnorm(DATA_TYPE out[MODEL_DIM],
                                DATA_TYPE x[MODEL_DIM],
                                DATA_TYPE weight[MODEL_DIM]);
void kernel_llama_lm_head_projection(DATA_TYPE x[MODEL_DIM],
                                     DATA_TYPE w[VOCAB][MODEL_DIM],
                                     DATA_TYPE logits[VOCAB]);

static DATA_TYPE g_embedding[VOCAB][MODEL_DIM];
static DATA_TYPE g_x[MODEL_DIM];
static DATA_TYPE g_residual[MODEL_DIM];
static DATA_TYPE g_weight[MODEL_DIM];
static DATA_TYPE g_w_model[MODEL_DIM][MODEL_DIM];
static DATA_TYPE g_wq[MODEL_DIM][MODEL_DIM];
static DATA_TYPE g_wk[MODEL_DIM][MODEL_DIM];
static DATA_TYPE g_wv[MODEL_DIM][MODEL_DIM];
static DATA_TYPE g_w_gate[FFN_DIM][MODEL_DIM];
static DATA_TYPE g_w_up[FFN_DIM][MODEL_DIM];
static DATA_TYPE g_w_down[MODEL_DIM][FFN_DIM];
static DATA_TYPE g_w_vocab[VOCAB][MODEL_DIM];
static DATA_TYPE g_q[MODEL_DIM];
static DATA_TYPE g_k[MODEL_DIM];
static DATA_TYPE g_v[MODEL_DIM];
static DATA_TYPE g_q_even[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_q_odd[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_k_even[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_k_odd[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_cos[SEQ_LEN][HALF_HEAD_DIM];
static DATA_TYPE g_sin[SEQ_LEN][HALF_HEAD_DIM];
static DATA_TYPE g_k_cache[SEQ_LEN][MODEL_DIM];
static DATA_TYPE g_v_cache[SEQ_LEN][MODEL_DIM];
static DATA_TYPE g_k_read[SEQ_LEN][MODEL_DIM];
static DATA_TYPE g_v_read[SEQ_LEN][MODEL_DIM];
static DATA_TYPE g_scores[SEQ_LEN];
static DATA_TYPE g_probs[SEQ_LEN];
static DATA_TYPE g_gate[FFN_DIM];
static DATA_TYPE g_up[FFN_DIM];
static DATA_TYPE g_hidden[FFN_DIM];
static DATA_TYPE g_out[MODEL_DIM];
static DATA_TYPE g_out2[MODEL_DIM];
static DATA_TYPE g_logits[VOCAB];
static DATA_TYPE g_q_even_out[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_q_odd_out[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_k_even_out[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE g_k_odd_out[NUM_HEADS][HALF_HEAD_DIM];

static DATA_TYPE init_value(int i, int j) {
  int v = (i * 17 + j * 13 + 7) % 101;
  return (DATA_TYPE)((v - 50) * 0.01f);
}

static void init_data(void) {
  for (int i = 0; i < VOCAB; ++i) {
    for (int j = 0; j < MODEL_DIM; ++j) {
      g_embedding[i][j] = init_value(i, j);
      g_w_vocab[i][j] = init_value(i + 3, j + 5);
    }
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    g_x[i] = init_value(i, 1);
    g_residual[i] = init_value(i, 2);
    g_weight[i] = (DATA_TYPE)1 + init_value(i, 3) * (DATA_TYPE)0.1;
    g_q[i] = init_value(i, 4);
    g_k[i] = init_value(i, 5);
    g_v[i] = init_value(i, 6);
    g_out[i] = (DATA_TYPE)0;
    g_out2[i] = (DATA_TYPE)0;
    for (int j = 0; j < MODEL_DIM; ++j) {
      g_w_model[i][j] = init_value(i, j);
      g_wq[i][j] = init_value(i + 1, j);
      g_wk[i][j] = init_value(i + 2, j);
      g_wv[i][j] = init_value(i + 3, j);
    }
    for (int j = 0; j < FFN_DIM; ++j) {
      g_w_down[i][j] = init_value(i, j + 4);
    }
  }
  for (int i = 0; i < FFN_DIM; ++i) {
    g_gate[i] = init_value(i, 7);
    g_up[i] = init_value(i, 8);
    g_hidden[i] = init_value(i, 9);
    for (int j = 0; j < MODEL_DIM; ++j) {
      g_w_gate[i][j] = init_value(i + 4, j);
      g_w_up[i][j] = init_value(i + 5, j);
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int p = 0; p < HALF_HEAD_DIM; ++p) {
      g_q_even[h][p] = init_value(h, p);
      g_q_odd[h][p] = init_value(h + 1, p);
      g_k_even[h][p] = init_value(h + 2, p);
      g_k_odd[h][p] = init_value(h + 3, p);
    }
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    g_scores[t] = init_value(t, 10);
    g_probs[t] = (DATA_TYPE)1 / (DATA_TYPE)SEQ_LEN;
    for (int p = 0; p < HALF_HEAD_DIM; ++p) {
      g_cos[t][p] = (DATA_TYPE)0.95 + (DATA_TYPE)0.001 * (DATA_TYPE)((t + p) % 7);
      g_sin[t][p] = (DATA_TYPE)0.05 + (DATA_TYPE)0.001 * (DATA_TYPE)((t + p) % 5);
    }
    for (int i = 0; i < MODEL_DIM; ++i) {
      g_k_cache[t][i] = init_value(t, i);
      g_v_cache[t][i] = init_value(t + 1, i);
      g_k_read[t][i] = (DATA_TYPE)0;
      g_v_read[t][i] = (DATA_TYPE)0;
    }
  }
}

static double checksum_1d(const DATA_TYPE *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += (double)x[i] * (double)(i + 1);
  }
  return s;
}

static double checksum_2d(const DATA_TYPE *x, int rows, int cols) {
  double s = 0.0;
  for (int i = 0; i < rows * cols; ++i) {
    s += (double)x[i] * (double)((i % 17) + 1);
  }
  return s;
}

int main(void) {
  init_data();
  const int token = 7;
  const int pos = SEQ_LEN / 2;

  for (int rep = 0; rep < REPEAT; ++rep) {
#if LLAMA_OP == 1
    kernel_llama_token_embedding(token, g_embedding, g_out);
#elif LLAMA_OP == 2
    kernel_llama_attention_rmsnorm(g_out, g_x, g_weight);
#elif LLAMA_OP == 3
    kernel_llama_qkv_projection(g_x, g_wq, g_wk, g_wv, g_q, g_k, g_v);
#elif LLAMA_OP == 4
    kernel_llama_rope_split(pos, g_q_even, g_q_odd, g_k_even, g_k_odd,
                            g_cos, g_sin, g_q_even_out, g_q_odd_out,
                            g_k_even_out, g_k_odd_out);
#elif LLAMA_OP == 5
    kernel_llama_kv_cache_rw(pos, g_k, g_v, g_k_cache, g_v_cache,
                             g_k_read, g_v_read);
#elif LLAMA_OP == 6
    kernel_llama_attention_scores(g_q, g_k_cache, g_scores);
#elif LLAMA_OP == 7
    kernel_llama_attention_mask_select(pos, g_scores, g_out);
#elif LLAMA_OP == 8
    kernel_llama_attention_softmax(g_probs, g_scores);
#elif LLAMA_OP == 9
    kernel_llama_attention_output(g_probs, g_v_cache, g_out);
#elif LLAMA_OP == 10
    kernel_llama_output_projection(g_x, g_w_model, g_out);
#elif LLAMA_OP == 11
    kernel_llama_residual_add(g_out, g_x, g_residual);
#elif LLAMA_OP == 12
    kernel_llama_ffn_rmsnorm(g_out, g_x, g_weight);
#elif LLAMA_OP == 13
    kernel_llama_gate_up_projection(g_x, g_w_gate, g_w_up, g_gate, g_up);
#elif LLAMA_OP == 14
    kernel_llama_swiglu(g_gate, g_up, g_hidden);
#elif LLAMA_OP == 15
    kernel_llama_down_projection(g_hidden, g_w_down, g_out);
#elif LLAMA_OP == 16
    kernel_llama_final_rmsnorm(g_out, g_x, g_weight);
#elif LLAMA_OP == 17
    kernel_llama_lm_head_projection(g_x, g_w_vocab, g_logits);
#else
#error "Unknown LLAMA_OP"
#endif
  }

  double checksum = 0.0;
  checksum += checksum_1d(g_out, MODEL_DIM);
  checksum += checksum_1d(g_out2, MODEL_DIM);
  checksum += checksum_1d(g_q, MODEL_DIM);
  checksum += checksum_1d(g_k, MODEL_DIM);
  checksum += checksum_1d(g_v, MODEL_DIM);
  checksum += checksum_1d(g_probs, SEQ_LEN);
  checksum += checksum_1d(g_hidden, FFN_DIM);
  checksum += checksum_1d(g_logits, VOCAB);
  checksum += checksum_2d(&g_k_read[0][0], SEQ_LEN, MODEL_DIM);
  checksum += checksum_2d(&g_v_read[0][0], SEQ_LEN, MODEL_DIM);
  checksum += checksum_2d(&g_q_even_out[0][0], NUM_HEADS, HALF_HEAD_DIM);
  checksum += checksum_2d(&g_q_odd_out[0][0], NUM_HEADS, HALF_HEAD_DIM);
  checksum += checksum_2d(&g_k_even_out[0][0], NUM_HEADS, HALF_HEAD_DIM);
  checksum += checksum_2d(&g_k_odd_out[0][0], NUM_HEADS, HALF_HEAD_DIM);
  printf("LLAMA_OP=%d checksum=%.9f\n", LLAMA_OP, checksum);
  return 0;
}
