/* llama_forward_ops.c -- standalone Llama-forward operation fixtures.
 *
 * Each function isolates one transformer-forward component so we can ask a
 * narrow question: does this C loop shape raise to linalg, and can the raised
 * memref form be debufferized to tensor linalg?
 */

#include <math.h>

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

#define NEG_INF ((DATA_TYPE)-3.4028234663852886e38f)


void kernel_llama_token_embedding(int token,
                             DATA_TYPE embedding[VOCAB][MODEL_DIM],
                             DATA_TYPE out[MODEL_DIM]) {
#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = embedding[token][i];
  }
#pragma endscop
}

void kernel_llama_attention_rmsnorm(DATA_TYPE out[MODEL_DIM],
                               DATA_TYPE x[MODEL_DIM],
                               DATA_TYPE weight[MODEL_DIM]) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    ss += x[i] * x[i];
  }
  ss /= (DATA_TYPE)MODEL_DIM;
  ss += (DATA_TYPE)1.0e-5;
  ss = (DATA_TYPE)1 / sqrtf(ss);
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = weight[i] * (ss * x[i]);
  }
#pragma endscop
}

void kernel_llama_qkv_projection(DATA_TYPE x[MODEL_DIM],
                            DATA_TYPE wq[MODEL_DIM][MODEL_DIM],
                            DATA_TYPE wk[MODEL_DIM][MODEL_DIM],
                            DATA_TYPE wv[MODEL_DIM][MODEL_DIM],
                            DATA_TYPE q[MODEL_DIM],
                            DATA_TYPE k[MODEL_DIM],
                            DATA_TYPE v[MODEL_DIM]) {
#pragma scop
  for (int row = 0; row < MODEL_DIM; ++row) {
    q[row] = (DATA_TYPE)0;
    k[row] = (DATA_TYPE)0;
    v[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      q[row] += wq[row][col] * x[col];
      k[row] += wk[row][col] * x[col];
      v[row] += wv[row][col] * x[col];
    }
  }
#pragma endscop
}

void kernel_llama_rope(int pos, DATA_TYPE q[NUM_HEADS][HEAD_DIM],
                  DATA_TYPE k[NUM_HEADS][HEAD_DIM],
                  DATA_TYPE cos_table[SEQ_LEN][HALF_HEAD_DIM],
                  DATA_TYPE sin_table[SEQ_LEN][HALF_HEAD_DIM],
                  DATA_TYPE q_out[NUM_HEADS][HEAD_DIM],
                  DATA_TYPE k_out[NUM_HEADS][HEAD_DIM]) {
#pragma scop
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      int even = 2 * pair;
      int odd = even + 1;
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      DATA_TYPE q_even = q[h][even];
      DATA_TYPE q_odd = q[h][odd];
      DATA_TYPE k_even = k[h][even];
      DATA_TYPE k_odd = k[h][odd];

      q_out[h][even] = q_even * c - q_odd * s;
      q_out[h][odd] = q_even * s + q_odd * c;
      k_out[h][even] = k_even * c - k_odd * s;
      k_out[h][odd] = k_even * s + k_odd * c;
    }
  }
#pragma endscop
}

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
                        DATA_TYPE k_odd_out[NUM_HEADS][HALF_HEAD_DIM]) {
#pragma scop
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      q_even_out[h][pair] = q_even[h][pair] * c - q_odd[h][pair] * s;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      q_odd_out[h][pair] = q_even[h][pair] * s + q_odd[h][pair] * c;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      k_even_out[h][pair] = k_even[h][pair] * c - k_odd[h][pair] * s;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      k_odd_out[h][pair] = k_even[h][pair] * s + k_odd[h][pair] * c;
    }
  }
#pragma endscop
}

void kernel_llama_kv_cache_rw(int pos, DATA_TYPE k[MODEL_DIM],
                         DATA_TYPE v[MODEL_DIM],
                         DATA_TYPE k_cache[SEQ_LEN][MODEL_DIM],
                         DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM],
                         DATA_TYPE k_read[SEQ_LEN][MODEL_DIM],
                         DATA_TYPE v_read[SEQ_LEN][MODEL_DIM]) {
#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    k_cache[pos][i] = k[i];
    v_cache[pos][i] = v[i];
  }

  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int i = 0; i < MODEL_DIM; ++i) {
      k_read[t][i] = k_cache[t][i];
      v_read[t][i] = v_cache[t][i];
    }
  }
#pragma endscop
}

void kernel_llama_attention_scores(DATA_TYPE q[MODEL_DIM],
                              DATA_TYPE k_cache[SEQ_LEN][MODEL_DIM],
                              DATA_TYPE scores[SEQ_LEN]) {
#pragma scop
  for (int t = 0; t < SEQ_LEN; ++t) {
    scores[t] = (DATA_TYPE)0;
  }

  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int i = 0; i < MODEL_DIM; ++i) {
      scores[t] += q[i] * k_cache[t][i];
    }
  }
#pragma endscop
}

void kernel_llama_attention_mask(int pos, DATA_TYPE scores[SEQ_LEN],
                            DATA_TYPE masked[SEQ_LEN]) {
#pragma scop
  for (int t = 0; t < SEQ_LEN; ++t) {
    if (t > pos) {
      masked[t] = NEG_INF;
    } else {
      masked[t] = scores[t];
    }
  }
#pragma endscop
}

void kernel_llama_attention_mask_select(int pos, DATA_TYPE scores[SEQ_LEN],
                                   DATA_TYPE masked[SEQ_LEN]) {
#pragma scop
  for (int t = 0; t < SEQ_LEN; ++t) {
    DATA_TYPE drop = (DATA_TYPE)(t > pos);
    DATA_TYPE keep = (DATA_TYPE)1 - drop;
    masked[t] = keep * scores[t] + drop * NEG_INF;
  }
#pragma endscop
}

void kernel_llama_attention_softmax(DATA_TYPE out[SEQ_LEN],
                               DATA_TYPE scores[SEQ_LEN]) {
  DATA_TYPE max_val = scores[0];

#pragma scop
  for (int t = 1; t < SEQ_LEN; ++t) {
    if (scores[t] > max_val) {
      max_val = scores[t];
    }
  }

  DATA_TYPE sum = (DATA_TYPE)0;
  for (int t = 0; t < SEQ_LEN; ++t) {
    out[t] = expf(scores[t] - max_val);
    sum += out[t];
  }

  for (int t = 0; t < SEQ_LEN; ++t) {
    out[t] /= sum;
  }
#pragma endscop
}

void kernel_llama_attention_output(DATA_TYPE probs[SEQ_LEN],
                              DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM],
                              DATA_TYPE out[MODEL_DIM]) {
#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = (DATA_TYPE)0;
  }

  for (int i = 0; i < MODEL_DIM; ++i) {
    for (int t = 0; t < SEQ_LEN; ++t) {
      out[i] += probs[t] * v_cache[t][i];
    }
  }
#pragma endscop
}

void kernel_llama_output_projection(DATA_TYPE x[MODEL_DIM],
                               DATA_TYPE w[MODEL_DIM][MODEL_DIM],
                               DATA_TYPE out[MODEL_DIM]) {
#pragma scop
  for (int row = 0; row < MODEL_DIM; ++row) {
    out[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      out[row] += w[row][col] * x[col];
    }
  }
#pragma endscop
}

void kernel_llama_residual_add(DATA_TYPE out[MODEL_DIM],
                          DATA_TYPE x[MODEL_DIM],
                          DATA_TYPE residual[MODEL_DIM]) {
#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = x[i] + residual[i];
  }
#pragma endscop
}

void kernel_llama_ffn_rmsnorm(DATA_TYPE out[MODEL_DIM],
                         DATA_TYPE x[MODEL_DIM],
                         DATA_TYPE weight[MODEL_DIM]) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    ss += x[i] * x[i];
  }
  ss /= (DATA_TYPE)MODEL_DIM;
  ss += (DATA_TYPE)1.0e-5;
  ss = (DATA_TYPE)1 / sqrtf(ss);
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = weight[i] * (ss * x[i]);
  }
#pragma endscop
}

void kernel_llama_gate_up_projection(DATA_TYPE x[MODEL_DIM],
                                DATA_TYPE w_gate[FFN_DIM][MODEL_DIM],
                                DATA_TYPE w_up[FFN_DIM][MODEL_DIM],
                                DATA_TYPE gate[FFN_DIM],
                                DATA_TYPE up[FFN_DIM]) {
#pragma scop
  for (int row = 0; row < FFN_DIM; ++row) {
    gate[row] = (DATA_TYPE)0;
    up[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < FFN_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      gate[row] += w_gate[row][col] * x[col];
      up[row] += w_up[row][col] * x[col];
    }
  }
#pragma endscop
}

void kernel_llama_swiglu(DATA_TYPE gate[FFN_DIM], DATA_TYPE up[FFN_DIM],
                    DATA_TYPE out[FFN_DIM]) {
#pragma scop
  for (int i = 0; i < FFN_DIM; ++i) {
    DATA_TYPE g = gate[i];
    DATA_TYPE silu = g / ((DATA_TYPE)1 + expf(-g));
    out[i] = silu * up[i];
  }
#pragma endscop
}

void kernel_llama_down_projection(DATA_TYPE hidden[FFN_DIM],
                             DATA_TYPE w[MODEL_DIM][FFN_DIM],
                             DATA_TYPE out[MODEL_DIM]) {
#pragma scop
  for (int row = 0; row < MODEL_DIM; ++row) {
    out[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < FFN_DIM; ++col) {
      out[row] += w[row][col] * hidden[col];
    }
  }
#pragma endscop
}

void kernel_llama_final_rmsnorm(DATA_TYPE out[MODEL_DIM],
                           DATA_TYPE x[MODEL_DIM],
                           DATA_TYPE weight[MODEL_DIM]) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    ss += x[i] * x[i];
  }
  ss /= (DATA_TYPE)MODEL_DIM;
  ss += (DATA_TYPE)1.0e-5;
  ss = (DATA_TYPE)1 / sqrtf(ss);
  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = weight[i] * (ss * x[i]);
  }
#pragma endscop
}

void kernel_llama_lm_head_projection(DATA_TYPE x[MODEL_DIM],
                                DATA_TYPE w[VOCAB][MODEL_DIM],
                                DATA_TYPE logits[VOCAB]) {
#pragma scop
  for (int row = 0; row < VOCAB; ++row) {
    logits[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < VOCAB; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      logits[row] += w[row][col] * x[col];
    }
  }
#pragma endscop
}
