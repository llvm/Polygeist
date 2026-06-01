/* llama2_extended_forward_bench.c -- fuller Llama2-style decode fixture.
 *
 * This is still a benchmark fixture, not the full Karpathy runtime. It models
 * one token through one transformer block plus final logits:
 *
 *   token embedding
 *   attention RMSNorm
 *   Q/K/V projections
 *   split-layout RoPE
 *   KV cache write/read
 *   attention scores + causal mask + softmax
 *   attention value matvec + output projection + residual
 *   FFN RMSNorm + gate/up projections + SwiGLU + down projection + residual
 *   final RMSNorm + lm_head projection
 *
 * Two deliberate raise-friendly choices:
 *   1. Q/K and RoPE use split even/odd tensors because the exact interleaved
 *      layout is a known remaining raising gap.
 *   2. The causal mask uses a branchless select expression because the branchy
 *      if/else form is also a known raising gap.
 */

#include <math.h>
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

#ifndef REPEAT
#define REPEAT 1
#endif

#ifndef PRINT_ELEMS
#define PRINT_ELEMS 8
#endif

#define NEG_INF ((DATA_TYPE)-3.4028234663852886e38f)

__attribute__((noinline)) void kernel_llama2_extended_forward(
    int token, int pos,
    DATA_TYPE tok_embeddings[VOCAB][MODEL_DIM],
    DATA_TYPE rms_att_weight[MODEL_DIM],
    DATA_TYPE wq_even[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM],
    DATA_TYPE wq_odd[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM],
    DATA_TYPE wk_even[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM],
    DATA_TYPE wk_odd[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM],
    DATA_TYPE wv[MODEL_DIM][MODEL_DIM],
    DATA_TYPE wo[MODEL_DIM][MODEL_DIM],
    DATA_TYPE rms_ffn_weight[MODEL_DIM],
    DATA_TYPE w_gate[FFN_DIM][MODEL_DIM],
    DATA_TYPE w_up[FFN_DIM][MODEL_DIM],
    DATA_TYPE w_down[MODEL_DIM][FFN_DIM],
    DATA_TYPE rms_final_weight[MODEL_DIM],
    DATA_TYPE lm_head[VOCAB][MODEL_DIM],
    DATA_TYPE cos_table[SEQ_LEN][HALF_HEAD_DIM],
    DATA_TYPE sin_table[SEQ_LEN][HALF_HEAD_DIM],
    DATA_TYPE k_cache_even[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE k_cache_odd[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM],
    DATA_TYPE x[MODEL_DIM],
    DATA_TYPE att_normed[MODEL_DIM],
    DATA_TYPE v[MODEL_DIM],
    DATA_TYPE q_even[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE q_odd[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE k_even[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE k_odd[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE q_even_rot[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE q_odd_rot[NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE k_read_even[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE k_read_odd[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM],
    DATA_TYPE v_read[SEQ_LEN][MODEL_DIM],
    DATA_TYPE scores[SEQ_LEN],
    DATA_TYPE masked_scores[SEQ_LEN],
    DATA_TYPE probs[SEQ_LEN],
    DATA_TYPE att_out[MODEL_DIM],
    DATA_TYPE proj_out[MODEL_DIM],
    DATA_TYPE resid_att[MODEL_DIM],
    DATA_TYPE ffn_normed[MODEL_DIM],
    DATA_TYPE gate[FFN_DIM],
    DATA_TYPE up[FFN_DIM],
    DATA_TYPE ffn_hidden[FFN_DIM],
    DATA_TYPE ffn_out[MODEL_DIM],
    DATA_TYPE resid_ffn[MODEL_DIM],
    DATA_TYPE final_normed[MODEL_DIM],
    DATA_TYPE logits[VOCAB]) {
  DATA_TYPE ss_att = (DATA_TYPE)0;
  DATA_TYPE ss_ffn = (DATA_TYPE)0;
  DATA_TYPE ss_final = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < MODEL_DIM; ++i) {
    x[i] = tok_embeddings[token][i];
  }

  for (int i = 0; i < MODEL_DIM; ++i) {
    ss_att += x[i] * x[i];
  }
  ss_att /= (DATA_TYPE)MODEL_DIM;
  ss_att += (DATA_TYPE)1.0e-5;
  ss_att = (DATA_TYPE)1 / sqrtf(ss_att);
  for (int i = 0; i < MODEL_DIM; ++i) {
    att_normed[i] = rms_att_weight[i] * (ss_att * x[i]);
  }

  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      q_even[h][pair] = (DATA_TYPE)0;
      q_odd[h][pair] = (DATA_TYPE)0;
      k_even[h][pair] = (DATA_TYPE)0;
      k_odd[h][pair] = (DATA_TYPE)0;
    }
  }
  for (int row = 0; row < MODEL_DIM; ++row) {
    v[row] = (DATA_TYPE)0;
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      for (int col = 0; col < MODEL_DIM; ++col) {
        q_even[h][pair] += wq_even[h][pair][col] * att_normed[col];
        q_odd[h][pair] += wq_odd[h][pair][col] * att_normed[col];
        k_even[h][pair] += wk_even[h][pair][col] * att_normed[col];
        k_odd[h][pair] += wk_odd[h][pair][col] * att_normed[col];
      }
    }
  }
  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      v[row] += wv[row][col] * att_normed[col];
    }
  }

  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      q_even_rot[h][pair] = q_even[h][pair] * c - q_odd[h][pair] * s;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      q_odd_rot[h][pair] = q_even[h][pair] * s + q_odd[h][pair] * c;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      k_cache_even[pos][h][pair] = k_even[h][pair] * c - k_odd[h][pair] * s;
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
      DATA_TYPE c = cos_table[pos][pair];
      DATA_TYPE s = sin_table[pos][pair];
      k_cache_odd[pos][h][pair] = k_even[h][pair] * s + k_odd[h][pair] * c;
    }
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
        k_read_even[t][h][pair] = k_cache_even[t][h][pair];
        k_read_odd[t][h][pair] = k_cache_odd[t][h][pair];
      }
    }
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    v_cache[pos][i] = v[i];
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int i = 0; i < MODEL_DIM; ++i) {
      v_read[t][i] = v_cache[t][i];
    }
  }

  for (int t = 0; t < SEQ_LEN; ++t) {
    scores[t] = (DATA_TYPE)0;
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int pair = 0; pair < HALF_HEAD_DIM; ++pair) {
        scores[t] += q_even_rot[h][pair] * k_read_even[t][h][pair] +
                     q_odd_rot[h][pair] * k_read_odd[t][h][pair];
      }
    }
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    scores[t] /= sqrtf((DATA_TYPE)HEAD_DIM);
  }

  for (int t = 0; t < SEQ_LEN; ++t) {
    DATA_TYPE drop = (DATA_TYPE)(t > pos);
    DATA_TYPE keep = (DATA_TYPE)1 - drop;
    masked_scores[t] = keep * scores[t] + drop * NEG_INF;
  }

  DATA_TYPE max_val = masked_scores[0];
  for (int t = 1; t < SEQ_LEN; ++t) {
    if (masked_scores[t] > max_val) {
      max_val = masked_scores[t];
    }
  }
  DATA_TYPE sum = (DATA_TYPE)0;
  for (int t = 0; t < SEQ_LEN; ++t) {
    probs[t] = expf(masked_scores[t] - max_val);
    sum += probs[t];
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    probs[t] /= sum;
  }

  for (int i = 0; i < MODEL_DIM; ++i) {
    att_out[i] = (DATA_TYPE)0;
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    for (int t = 0; t < SEQ_LEN; ++t) {
      att_out[i] += probs[t] * v_read[t][i];
    }
  }

  for (int row = 0; row < MODEL_DIM; ++row) {
    proj_out[row] = (DATA_TYPE)0;
  }
  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      proj_out[row] += wo[row][col] * att_out[col];
    }
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    resid_att[i] = x[i] + proj_out[i];
  }

  for (int i = 0; i < MODEL_DIM; ++i) {
    ss_ffn += resid_att[i] * resid_att[i];
  }
  ss_ffn /= (DATA_TYPE)MODEL_DIM;
  ss_ffn += (DATA_TYPE)1.0e-5;
  ss_ffn = (DATA_TYPE)1 / sqrtf(ss_ffn);
  for (int i = 0; i < MODEL_DIM; ++i) {
    ffn_normed[i] = rms_ffn_weight[i] * (ss_ffn * resid_att[i]);
  }

  for (int row = 0; row < FFN_DIM; ++row) {
    gate[row] = (DATA_TYPE)0;
    up[row] = (DATA_TYPE)0;
  }
  for (int row = 0; row < FFN_DIM; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      gate[row] += w_gate[row][col] * ffn_normed[col];
      up[row] += w_up[row][col] * ffn_normed[col];
    }
  }
  for (int i = 0; i < FFN_DIM; ++i) {
    DATA_TYPE g = gate[i];
    DATA_TYPE silu = g / ((DATA_TYPE)1 + expf(-g));
    ffn_hidden[i] = silu * up[i];
  }

  for (int row = 0; row < MODEL_DIM; ++row) {
    ffn_out[row] = (DATA_TYPE)0;
  }
  for (int row = 0; row < MODEL_DIM; ++row) {
    for (int col = 0; col < FFN_DIM; ++col) {
      ffn_out[row] += w_down[row][col] * ffn_hidden[col];
    }
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    resid_ffn[i] = resid_att[i] + ffn_out[i];
  }

  for (int i = 0; i < MODEL_DIM; ++i) {
    ss_final += resid_ffn[i] * resid_ffn[i];
  }
  ss_final /= (DATA_TYPE)MODEL_DIM;
  ss_final += (DATA_TYPE)1.0e-5;
  ss_final = (DATA_TYPE)1 / sqrtf(ss_final);
  for (int i = 0; i < MODEL_DIM; ++i) {
    final_normed[i] = rms_final_weight[i] * (ss_final * resid_ffn[i]);
  }

  for (int row = 0; row < VOCAB; ++row) {
    logits[row] = (DATA_TYPE)0;
  }
  for (int row = 0; row < VOCAB; ++row) {
    for (int col = 0; col < MODEL_DIM; ++col) {
      logits[row] += lm_head[row][col] * final_normed[col];
    }
  }
#pragma endscop
}

static DATA_TYPE tok_embeddings[VOCAB][MODEL_DIM];
static DATA_TYPE rms_att_weight[MODEL_DIM];
static DATA_TYPE wq_even[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM];
static DATA_TYPE wq_odd[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM];
static DATA_TYPE wk_even[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM];
static DATA_TYPE wk_odd[NUM_HEADS][HALF_HEAD_DIM][MODEL_DIM];
static DATA_TYPE wv[MODEL_DIM][MODEL_DIM];
static DATA_TYPE wo[MODEL_DIM][MODEL_DIM];
static DATA_TYPE rms_ffn_weight[MODEL_DIM];
static DATA_TYPE w_gate[FFN_DIM][MODEL_DIM];
static DATA_TYPE w_up[FFN_DIM][MODEL_DIM];
static DATA_TYPE w_down[MODEL_DIM][FFN_DIM];
static DATA_TYPE rms_final_weight[MODEL_DIM];
static DATA_TYPE lm_head[VOCAB][MODEL_DIM];
static DATA_TYPE cos_table[SEQ_LEN][HALF_HEAD_DIM];
static DATA_TYPE sin_table[SEQ_LEN][HALF_HEAD_DIM];
static DATA_TYPE k_cache_even[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE k_cache_odd[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE v_cache[SEQ_LEN][MODEL_DIM];
static DATA_TYPE x[MODEL_DIM];
static DATA_TYPE att_normed[MODEL_DIM];
static DATA_TYPE v[MODEL_DIM];
static DATA_TYPE q_even[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE q_odd[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE k_even[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE k_odd[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE q_even_rot[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE q_odd_rot[NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE k_read_even[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE k_read_odd[SEQ_LEN][NUM_HEADS][HALF_HEAD_DIM];
static DATA_TYPE v_read[SEQ_LEN][MODEL_DIM];
static DATA_TYPE scores[SEQ_LEN];
static DATA_TYPE masked_scores[SEQ_LEN];
static DATA_TYPE probs[SEQ_LEN];
static DATA_TYPE att_out[MODEL_DIM];
static DATA_TYPE proj_out[MODEL_DIM];
static DATA_TYPE resid_att[MODEL_DIM];
static DATA_TYPE ffn_normed[MODEL_DIM];
static DATA_TYPE gate[FFN_DIM];
static DATA_TYPE up[FFN_DIM];
static DATA_TYPE ffn_hidden[FFN_DIM];
static DATA_TYPE ffn_out[MODEL_DIM];
static DATA_TYPE resid_ffn[MODEL_DIM];
static DATA_TYPE final_normed[MODEL_DIM];
static DATA_TYPE logits[VOCAB];

static DATA_TYPE init_value(int i, int j) {
  int v = (i * 17 + j * 13 + 7) % 101;
  return (DATA_TYPE)((v - 50) * 0.01f);
}

static void init_array(void) {
  for (int i = 0; i < VOCAB; ++i) {
    for (int j = 0; j < MODEL_DIM; ++j) {
      tok_embeddings[i][j] = init_value(i, j);
      lm_head[i][j] = init_value(i + 3, j + 5);
    }
  }
  for (int i = 0; i < MODEL_DIM; ++i) {
    rms_att_weight[i] = (DATA_TYPE)1 + init_value(i, 1) * (DATA_TYPE)0.1;
    rms_ffn_weight[i] = (DATA_TYPE)1 + init_value(i, 2) * (DATA_TYPE)0.1;
    rms_final_weight[i] = (DATA_TYPE)1 + init_value(i, 3) * (DATA_TYPE)0.1;
    for (int j = 0; j < MODEL_DIM; ++j) {
      wv[i][j] = init_value(i + 3, j);
      wo[i][j] = init_value(i + 4, j);
    }
    for (int j = 0; j < FFN_DIM; ++j) {
      w_down[i][j] = init_value(i + 5, j);
    }
  }
  for (int i = 0; i < FFN_DIM; ++i) {
    for (int j = 0; j < MODEL_DIM; ++j) {
      w_gate[i][j] = init_value(i + 6, j);
      w_up[i][j] = init_value(i + 7, j);
    }
  }
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int p = 0; p < HALF_HEAD_DIM; ++p) {
      int row_even = h * HEAD_DIM + 2 * p;
      int row_odd = row_even + 1;
      for (int j = 0; j < MODEL_DIM; ++j) {
        wq_even[h][p][j] = init_value(row_even + 1, j);
        wq_odd[h][p][j] = init_value(row_odd + 1, j);
        wk_even[h][p][j] = init_value(row_even + 2, j);
        wk_odd[h][p][j] = init_value(row_odd + 2, j);
      }
    }
  }
  for (int t = 0; t < SEQ_LEN; ++t) {
    for (int p = 0; p < HALF_HEAD_DIM; ++p) {
      cos_table[t][p] = (DATA_TYPE)0.95 +
                        (DATA_TYPE)0.001 * (DATA_TYPE)((t + p) % 7);
      sin_table[t][p] = (DATA_TYPE)0.05 +
                        (DATA_TYPE)0.001 * (DATA_TYPE)((t + p) % 5);
    }
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int p = 0; p < HALF_HEAD_DIM; ++p) {
        k_cache_even[t][h][p] = init_value(t + h, p);
        k_cache_odd[t][h][p] = init_value(t + h + 1, p);
      }
    }
    for (int i = 0; i < MODEL_DIM; ++i) {
      v_cache[t][i] = init_value(t + 1, i);
    }
  }
}

static void print_array(void) {
  int nprint = PRINT_ELEMS < VOCAB ? PRINT_ELEMS : VOCAB;
  DATA_TYPE checksum = (DATA_TYPE)0;
  for (int i = 0; i < VOCAB; ++i) {
    checksum += logits[i];
  }
  for (int i = 0; i < nprint; ++i) {
    printf("%.8f\n", (double)logits[i]);
  }
  printf("%.8f\n", (double)checksum);
}

int main(void) {
  const int token = 7;
  const int pos = SEQ_LEN / 2;
  init_array();
  for (int r = 0; r < REPEAT; ++r) {
    kernel_llama2_extended_forward(
        token, pos, tok_embeddings, rms_att_weight, wq_even, wq_odd, wk_even,
        wk_odd, wv, wo, rms_ffn_weight, w_gate, w_up, w_down,
        rms_final_weight, lm_head, cos_table, sin_table, k_cache_even,
        k_cache_odd, v_cache, x, att_normed, v, q_even, q_odd, k_even, k_odd,
        q_even_rot, q_odd_rot, k_read_even, k_read_odd, v_read, scores,
        masked_scores, probs, att_out, proj_out, resid_att, ffn_normed, gate,
        up, ffn_hidden, ffn_out, resid_ffn, final_normed, logits);
  }
  print_array();
  return 0;
}
