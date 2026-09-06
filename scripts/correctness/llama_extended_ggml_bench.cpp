// ggml/CUDA benchmark for the same full Llama-style fixture as:
//
//   third_party/cnn-extracted/llama2_extended_forward_bench.c
//
// This intentionally mirrors that f32 fixture, including its split even/odd
// Q/K layout and branchless causal mask. It is not a GGUF/TinyLlama runner.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#define NEG_INF (-3.4028234663852886e38f)

namespace {

struct Options {
    int warmup = 0;
    int iters = 1;
    int token = 7;
    int pos = SEQ_LEN / 2;
    std::string stage = "logits";
    bool dump_all = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--warmup W] [--iters I] [--token T] [--pos P] "
                 "[--stage x|att_normed|q_even|k_even|scores|probs|att_out|"
                 "resid_att|ffn_hidden|resid_ffn|final_normed|logits] "
                 "[--dump-all]\n",
                 argv0);
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value > 2147483647L) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

static Options parse_options(int argc, char ** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        int * target = nullptr;
        if (arg == "--warmup") {
            target = &opts.warmup;
        } else if (arg == "--iters") {
            target = &opts.iters;
        } else if (arg == "--token") {
            target = &opts.token;
        } else if (arg == "--pos") {
            target = &opts.pos;
        } else if (arg == "--stage") {
            if (++i >= argc) {
                usage(argv[0]);
                std::exit(2);
            }
            opts.stage = argv[i];
            if (opts.stage != "x" && opts.stage != "att_normed" &&
                opts.stage != "q_even" && opts.stage != "k_even" &&
                opts.stage != "scores" && opts.stage != "probs" &&
                opts.stage != "att_out" && opts.stage != "resid_att" &&
                opts.stage != "ffn_hidden" && opts.stage != "resid_ffn" &&
                opts.stage != "final_normed" && opts.stage != "logits") {
                usage(argv[0]);
                std::exit(2);
            }
            continue;
        } else if (arg == "--dump-all") {
            opts.dump_all = true;
            continue;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            std::exit(2);
        }

        if (++i >= argc || !parse_int(argv[i], *target)) {
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (opts.warmup < 0 || opts.iters <= 0 || opts.token < 0 ||
        opts.token >= VOCAB || opts.pos < 0 || opts.pos >= SEQ_LEN) {
        usage(argv[0]);
        std::exit(2);
    }
    return opts;
}

static float init_value(int i, int j) {
    int v = (i * 17 + j * 13 + 7) % 101;
    return static_cast<float>(v - 50) * 0.01f;
}

static double average(const std::vector<double> & xs) {
    double sum = 0.0;
    for (double x : xs) {
        sum += x;
    }
    return sum / static_cast<double>(xs.size());
}

static double median(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    const size_t mid = xs.size() / 2;
    if ((xs.size() & 1) != 0) {
        return xs[mid];
    }
    return 0.5 * (xs[mid - 1] + xs[mid]);
}

static double trimmed_mean(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    if (xs.size() <= 4) {
        return average(xs);
    }
    const size_t drop = std::max<size_t>(1, xs.size() / 10);
    double sum = 0.0;
    for (size_t i = drop; i < xs.size() - drop; ++i) {
        sum += xs[i];
    }
    return sum / static_cast<double>(xs.size() - 2 * drop);
}

struct Inputs {
    std::vector<float> tok_embeddings;
    std::vector<float> rms_att_weight;
    std::vector<float> wq_even;
    std::vector<float> wq_odd;
    std::vector<float> wk_even;
    std::vector<float> wk_odd;
    std::vector<float> wv;
    std::vector<float> wo;
    std::vector<float> rms_ffn_weight;
    std::vector<float> w_gate;
    std::vector<float> w_up;
    std::vector<float> w_down;
    std::vector<float> rms_final_weight;
    std::vector<float> lm_head;
    std::vector<float> cos_hp;
    std::vector<float> sin_hp;
    std::vector<float> mask;
    std::vector<float> k_cache_even;
    std::vector<float> k_cache_odd;
    std::vector<float> v_cache;
    int token = 7;
    int pos = SEQ_LEN / 2;
};

static void init_inputs(Inputs & in, int token, int pos) {
    constexpr int qk_rows = NUM_HEADS * HALF_HEAD_DIM;
    in.token = token;
    in.pos = pos;
    in.tok_embeddings.resize(static_cast<size_t>(VOCAB) * MODEL_DIM);
    in.rms_att_weight.resize(MODEL_DIM);
    in.wq_even.resize(static_cast<size_t>(qk_rows) * MODEL_DIM);
    in.wq_odd.resize(static_cast<size_t>(qk_rows) * MODEL_DIM);
    in.wk_even.resize(static_cast<size_t>(qk_rows) * MODEL_DIM);
    in.wk_odd.resize(static_cast<size_t>(qk_rows) * MODEL_DIM);
    in.wv.resize(static_cast<size_t>(MODEL_DIM) * MODEL_DIM);
    in.wo.resize(static_cast<size_t>(MODEL_DIM) * MODEL_DIM);
    in.rms_ffn_weight.resize(MODEL_DIM);
    in.w_gate.resize(static_cast<size_t>(FFN_DIM) * MODEL_DIM);
    in.w_up.resize(static_cast<size_t>(FFN_DIM) * MODEL_DIM);
    in.w_down.resize(static_cast<size_t>(MODEL_DIM) * FFN_DIM);
    in.rms_final_weight.resize(MODEL_DIM);
    in.lm_head.resize(static_cast<size_t>(VOCAB) * MODEL_DIM);
    in.cos_hp.resize(qk_rows);
    in.sin_hp.resize(qk_rows);
    in.mask.resize(SEQ_LEN);
    in.k_cache_even.resize(static_cast<size_t>(SEQ_LEN) * qk_rows);
    in.k_cache_odd.resize(static_cast<size_t>(SEQ_LEN) * qk_rows);
    in.v_cache.resize(static_cast<size_t>(SEQ_LEN) * MODEL_DIM);

    for (int i = 0; i < VOCAB; ++i) {
        for (int j = 0; j < MODEL_DIM; ++j) {
            in.tok_embeddings[static_cast<size_t>(i) * MODEL_DIM + j] =
                init_value(i, j);
            in.lm_head[static_cast<size_t>(i) * MODEL_DIM + j] =
                init_value(i + 3, j + 5);
        }
    }
    for (int i = 0; i < MODEL_DIM; ++i) {
        in.rms_att_weight[i] = 1.0f + init_value(i, 1) * 0.1f;
        in.rms_ffn_weight[i] = 1.0f + init_value(i, 2) * 0.1f;
        in.rms_final_weight[i] = 1.0f + init_value(i, 3) * 0.1f;
        for (int j = 0; j < MODEL_DIM; ++j) {
            in.wv[static_cast<size_t>(i) * MODEL_DIM + j] = init_value(i + 3, j);
            in.wo[static_cast<size_t>(i) * MODEL_DIM + j] = init_value(i + 4, j);
        }
        for (int j = 0; j < FFN_DIM; ++j) {
            in.w_down[static_cast<size_t>(i) * FFN_DIM + j] = init_value(i + 5, j);
        }
    }
    for (int i = 0; i < FFN_DIM; ++i) {
        for (int j = 0; j < MODEL_DIM; ++j) {
            in.w_gate[static_cast<size_t>(i) * MODEL_DIM + j] = init_value(i + 6, j);
            in.w_up[static_cast<size_t>(i) * MODEL_DIM + j] = init_value(i + 7, j);
        }
    }
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int p = 0; p < HALF_HEAD_DIM; ++p) {
            const int flat = h * HALF_HEAD_DIM + p;
            const int row_even = h * HEAD_DIM + 2 * p;
            const int row_odd = row_even + 1;
            const float c = 0.95f + 0.001f * static_cast<float>((pos + p) % 7);
            const float s = 0.05f + 0.001f * static_cast<float>((pos + p) % 5);
            in.cos_hp[flat] = c;
            in.sin_hp[flat] = s;
            for (int j = 0; j < MODEL_DIM; ++j) {
                const size_t idx = static_cast<size_t>(flat) * MODEL_DIM + j;
                in.wq_even[idx] = init_value(row_even + 1, j);
                in.wq_odd[idx] = init_value(row_odd + 1, j);
                in.wk_even[idx] = init_value(row_even + 2, j);
                in.wk_odd[idx] = init_value(row_odd + 2, j);
            }
        }
    }
    for (int t = 0; t < SEQ_LEN; ++t) {
        in.mask[t] = t > pos ? NEG_INF : 0.0f;
        for (int h = 0; h < NUM_HEADS; ++h) {
            for (int p = 0; p < HALF_HEAD_DIM; ++p) {
                const int flat = h * HALF_HEAD_DIM + p;
                in.k_cache_even[static_cast<size_t>(t) * qk_rows + flat] =
                    init_value(t + h, p);
                in.k_cache_odd[static_cast<size_t>(t) * qk_rows + flat] =
                    init_value(t + h + 1, p);
            }
        }
        for (int i = 0; i < MODEL_DIM; ++i) {
            in.v_cache[static_cast<size_t>(t) * MODEL_DIM + i] =
                init_value(t + 1, i);
        }
    }
}

struct Bench {
    Options opts;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_buf;
    ggml_cgraph * graph = nullptr;

    ggml_tensor * token = nullptr;
    ggml_tensor * tok_embeddings = nullptr;
    ggml_tensor * rms_att_weight = nullptr;
    ggml_tensor * wq_even = nullptr;
    ggml_tensor * wq_odd = nullptr;
    ggml_tensor * wk_even = nullptr;
    ggml_tensor * wk_odd = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * rms_ffn_weight = nullptr;
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_down = nullptr;
    ggml_tensor * rms_final_weight = nullptr;
    ggml_tensor * lm_head = nullptr;
    ggml_tensor * cos_hp = nullptr;
    ggml_tensor * sin_hp = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * k_cache_even = nullptr;
    ggml_tensor * k_cache_odd = nullptr;
    ggml_tensor * v_cache = nullptr;
    ggml_tensor * out = nullptr;
};

static void init_backend(Bench & bench) {
    ggml_backend_load_all();

    bench.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (bench.backend == nullptr) {
        bench.backend = ggml_backend_init_best();
    }
    bench.cpu_backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (bench.backend == nullptr || bench.cpu_backend == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml backends\n");
        std::exit(1);
    }

    ggml_backend_t backends[2] = {bench.backend, bench.cpu_backend};
    bench.sched =
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE,
                               false, true);
    if (bench.sched == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml backend scheduler\n");
        std::exit(1);
    }
}

static ggml_tensor * vec_matmul(ggml_context * ctx, ggml_tensor * vec,
                                ggml_tensor * matrix, int cols, int rows) {
    ggml_tensor * vec2 = ggml_reshape_2d(ctx, vec, cols, 1);
    ggml_tensor * mm = ggml_mul_mat(ctx, vec2, matrix);
    return ggml_reshape_1d(ctx, mm, rows);
}

static void build_graph(Bench & bench) {
    constexpr int qk_rows = NUM_HEADS * HALF_HEAD_DIM;
    const size_t buf_size =
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    bench.graph_buf.resize(buf_size);

    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/bench.graph_buf.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        std::exit(1);
    }

    bench.graph = ggml_new_graph(ctx);
    bench.token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    bench.tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, VOCAB);
    bench.rms_att_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MODEL_DIM);
    bench.wq_even = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, qk_rows);
    bench.wq_odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, qk_rows);
    bench.wk_even = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, qk_rows);
    bench.wk_odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, qk_rows);
    bench.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, MODEL_DIM);
    bench.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, MODEL_DIM);
    bench.rms_ffn_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MODEL_DIM);
    bench.w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, FFN_DIM);
    bench.w_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, FFN_DIM);
    bench.w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FFN_DIM, MODEL_DIM);
    bench.rms_final_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MODEL_DIM);
    bench.lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, VOCAB);
    bench.cos_hp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qk_rows);
    bench.sin_hp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qk_rows);
    bench.mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SEQ_LEN);
    bench.k_cache_even = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qk_rows, SEQ_LEN);
    bench.k_cache_odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qk_rows, SEQ_LEN);
    bench.v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MODEL_DIM, SEQ_LEN);

    ggml_tensor * x = ggml_reshape_1d(
        ctx, ggml_get_rows(ctx, bench.tok_embeddings, bench.token), MODEL_DIM);

    ggml_tensor * att_normed =
        ggml_mul(ctx, ggml_rms_norm(ctx, x, 1.0e-5f), bench.rms_att_weight);

    ggml_tensor * q_even = vec_matmul(ctx, att_normed, bench.wq_even, MODEL_DIM, qk_rows);
    ggml_tensor * q_odd = vec_matmul(ctx, att_normed, bench.wq_odd, MODEL_DIM, qk_rows);
    ggml_tensor * k_even = vec_matmul(ctx, att_normed, bench.wk_even, MODEL_DIM, qk_rows);
    ggml_tensor * k_odd = vec_matmul(ctx, att_normed, bench.wk_odd, MODEL_DIM, qk_rows);
    ggml_tensor * v = vec_matmul(ctx, att_normed, bench.wv, MODEL_DIM, MODEL_DIM);

    ggml_tensor * q_even_c = ggml_mul(ctx, q_even, bench.cos_hp);
    ggml_tensor * q_odd_s = ggml_mul(ctx, q_odd, bench.sin_hp);
    ggml_tensor * q_even_rot = ggml_sub(ctx, q_even_c, q_odd_s);
    ggml_tensor * q_even_s = ggml_mul(ctx, q_even, bench.sin_hp);
    ggml_tensor * q_odd_c = ggml_mul(ctx, q_odd, bench.cos_hp);
    ggml_tensor * q_odd_rot = ggml_add(ctx, q_even_s, q_odd_c);

    ggml_tensor * k_even_c = ggml_mul(ctx, k_even, bench.cos_hp);
    ggml_tensor * k_odd_s = ggml_mul(ctx, k_odd, bench.sin_hp);
    ggml_tensor * k_even_rot = ggml_sub(ctx, k_even_c, k_odd_s);
    ggml_tensor * k_even_s = ggml_mul(ctx, k_even, bench.sin_hp);
    ggml_tensor * k_odd_c = ggml_mul(ctx, k_odd, bench.cos_hp);
    ggml_tensor * k_odd_rot = ggml_add(ctx, k_even_s, k_odd_c);

    const size_t k_offset =
        static_cast<size_t>(bench.opts.pos) * qk_rows * sizeof(float);
    const size_t v_offset =
        static_cast<size_t>(bench.opts.pos) * MODEL_DIM * sizeof(float);
    ggml_tensor * k_cache_even =
        ggml_set_1d(ctx, bench.k_cache_even, k_even_rot, k_offset);
    ggml_tensor * k_cache_odd =
        ggml_set_1d(ctx, bench.k_cache_odd, k_odd_rot, k_offset);
    ggml_tensor * v_cache = ggml_set_1d(ctx, bench.v_cache, v, v_offset);

    ggml_tensor * q_even2 = ggml_reshape_2d(ctx, q_even_rot, qk_rows, 1);
    ggml_tensor * q_odd2 = ggml_reshape_2d(ctx, q_odd_rot, qk_rows, 1);
    ggml_tensor * scores_even =
        ggml_reshape_1d(ctx, ggml_mul_mat(ctx, q_even2, k_cache_even), SEQ_LEN);
    ggml_tensor * scores_odd =
        ggml_reshape_1d(ctx, ggml_mul_mat(ctx, q_odd2, k_cache_odd), SEQ_LEN);
    ggml_tensor * scores = ggml_scale(
        ctx, ggml_add(ctx, scores_even, scores_odd),
        1.0f / std::sqrt(static_cast<float>(HEAD_DIM)));
    ggml_tensor * masked_scores = ggml_add(ctx, scores, bench.mask);
    ggml_tensor * probs = ggml_soft_max(ctx, masked_scores);

    ggml_tensor * probs2 = ggml_reshape_2d(ctx, probs, SEQ_LEN, 1);
    ggml_tensor * v_cache_t =
        ggml_cont_2d(ctx, ggml_transpose(ctx, v_cache), SEQ_LEN, MODEL_DIM);
    ggml_tensor * att_out =
        ggml_reshape_1d(ctx, ggml_mul_mat(ctx, probs2, v_cache_t), MODEL_DIM);

    ggml_tensor * proj_out = vec_matmul(ctx, att_out, bench.wo, MODEL_DIM, MODEL_DIM);
    ggml_tensor * resid_att = ggml_add(ctx, x, proj_out);

    ggml_tensor * ffn_normed =
        ggml_mul(ctx, ggml_rms_norm(ctx, resid_att, 1.0e-5f), bench.rms_ffn_weight);
    ggml_tensor * gate = vec_matmul(ctx, ffn_normed, bench.w_gate, MODEL_DIM, FFN_DIM);
    ggml_tensor * up = vec_matmul(ctx, ffn_normed, bench.w_up, MODEL_DIM, FFN_DIM);
    ggml_tensor * ffn_hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * ffn_out = vec_matmul(ctx, ffn_hidden, bench.w_down, FFN_DIM, MODEL_DIM);
    ggml_tensor * resid_ffn = ggml_add(ctx, resid_att, ffn_out);

    ggml_tensor * final_normed =
        ggml_mul(ctx, ggml_rms_norm(ctx, resid_ffn, 1.0e-5f), bench.rms_final_weight);
    ggml_tensor * logits = vec_matmul(ctx, final_normed, bench.lm_head, MODEL_DIM, VOCAB);

    if (bench.opts.stage == "x") {
        bench.out = x;
    } else if (bench.opts.stage == "att_normed") {
        bench.out = att_normed;
    } else if (bench.opts.stage == "q_even") {
        bench.out = q_even;
    } else if (bench.opts.stage == "k_even") {
        bench.out = k_even;
    } else if (bench.opts.stage == "scores") {
        bench.out = scores;
    } else if (bench.opts.stage == "probs") {
        bench.out = probs;
    } else if (bench.opts.stage == "att_out") {
        bench.out = att_out;
    } else if (bench.opts.stage == "resid_att") {
        bench.out = resid_att;
    } else if (bench.opts.stage == "ffn_hidden") {
        bench.out = ffn_hidden;
    } else if (bench.opts.stage == "resid_ffn") {
        bench.out = resid_ffn;
    } else if (bench.opts.stage == "final_normed") {
        bench.out = final_normed;
    } else {
        bench.out = logits;
    }

    ggml_build_forward_expand(bench.graph, bench.out);
    ggml_free(ctx);
}

static void set_tensor_if_allocated(ggml_tensor * tensor, const void * data) {
    if (tensor != nullptr && tensor->buffer != nullptr) {
        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    }
}

static void load_inputs(Bench & bench, const Inputs & in) {
    ggml_backend_sched_reset(bench.sched);
    if (!ggml_backend_sched_alloc_graph(bench.sched, bench.graph)) {
        std::fprintf(stderr, "failed to allocate ggml graph\n");
        std::exit(1);
    }

    int32_t token = in.token;
    set_tensor_if_allocated(bench.token, &token);
    set_tensor_if_allocated(bench.tok_embeddings, in.tok_embeddings.data());
    set_tensor_if_allocated(bench.rms_att_weight, in.rms_att_weight.data());
    set_tensor_if_allocated(bench.wq_even, in.wq_even.data());
    set_tensor_if_allocated(bench.wq_odd, in.wq_odd.data());
    set_tensor_if_allocated(bench.wk_even, in.wk_even.data());
    set_tensor_if_allocated(bench.wk_odd, in.wk_odd.data());
    set_tensor_if_allocated(bench.wv, in.wv.data());
    set_tensor_if_allocated(bench.wo, in.wo.data());
    set_tensor_if_allocated(bench.rms_ffn_weight, in.rms_ffn_weight.data());
    set_tensor_if_allocated(bench.w_gate, in.w_gate.data());
    set_tensor_if_allocated(bench.w_up, in.w_up.data());
    set_tensor_if_allocated(bench.w_down, in.w_down.data());
    set_tensor_if_allocated(bench.rms_final_weight, in.rms_final_weight.data());
    set_tensor_if_allocated(bench.lm_head, in.lm_head.data());
    set_tensor_if_allocated(bench.cos_hp, in.cos_hp.data());
    set_tensor_if_allocated(bench.sin_hp, in.sin_hp.data());
    set_tensor_if_allocated(bench.mask, in.mask.data());
    set_tensor_if_allocated(bench.k_cache_even, in.k_cache_even.data());
    set_tensor_if_allocated(bench.k_cache_odd, in.k_cache_odd.data());
    set_tensor_if_allocated(bench.v_cache, in.v_cache.data());
}

static double run_once(Bench & bench) {
    const int64_t t0 = ggml_time_us();
    const ggml_status status =
        ggml_backend_sched_graph_compute(bench.sched, bench.graph);
    const int64_t t1 = ggml_time_us();
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml graph compute failed: %d\n",
                     static_cast<int>(status));
        std::exit(1);
    }
    return static_cast<double>(t1 - t0) / 1000.0;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_time_init();

    Bench bench;
    bench.opts = parse_options(argc, argv);

    Inputs inputs;
    init_inputs(inputs, bench.opts.token, bench.opts.pos);

    init_backend(bench);
    build_graph(bench);

    std::fprintf(stderr,
                 "backend=%s model_dim=%d ffn_dim=%d vocab=%d seq_len=%d "
                 "heads=%d token=%d pos=%d warmup=%d iters=%d stage=%s\n",
                 ggml_backend_name(bench.backend), MODEL_DIM, FFN_DIM, VOCAB,
                 SEQ_LEN, NUM_HEADS, bench.opts.token, bench.opts.pos,
                 bench.opts.warmup, bench.opts.iters, bench.opts.stage.c_str());

    for (int i = 0; i < bench.opts.warmup; ++i) {
        load_inputs(bench, inputs);
        (void)run_once(bench);
    }

    std::vector<double> times;
    times.reserve(bench.opts.iters);
    for (int i = 0; i < bench.opts.iters; ++i) {
        load_inputs(bench, inputs);
        times.push_back(run_once(bench));
    }

    std::vector<float> out(static_cast<size_t>(ggml_nelements(bench.out)));
    ggml_backend_tensor_get(bench.out, out.data(), 0, ggml_nbytes(bench.out));

    double checksum = 0.0;
    double sumsq = 0.0;
    double maxabs = 0.0;
    for (float v : out) {
        checksum += static_cast<double>(v);
        sumsq += static_cast<double>(v) * static_cast<double>(v);
        maxabs = std::max(maxabs, std::abs(static_cast<double>(v)));
    }

    std::printf("bench,stage,backend,model_dim,ffn_dim,vocab,seq_len,heads,token,pos,"
                "warmup,iters,avg_ms,median_ms,trimmed_ms,min_ms,max_ms,"
                "checksum,sumsq,maxabs,out0,out1,out2,out3,out4,out5,out6,out7\n");
    std::printf("ggml_extended,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.17g,%.17g,%.17g,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                bench.opts.stage.c_str(), ggml_backend_name(bench.backend),
                MODEL_DIM, FFN_DIM, VOCAB, SEQ_LEN, NUM_HEADS,
                bench.opts.token, bench.opts.pos, bench.opts.warmup,
                bench.opts.iters, average(times),
                median(times), trimmed_mean(times),
                *std::min_element(times.begin(), times.end()),
                *std::max_element(times.begin(), times.end()), checksum, sumsq,
                maxabs,
                out.size() > 0 ? out[0] : 0.0f,
                out.size() > 1 ? out[1] : 0.0f,
                out.size() > 2 ? out[2] : 0.0f,
                out.size() > 3 ? out[3] : 0.0f,
                out.size() > 4 ? out[4] : 0.0f,
                out.size() > 5 ? out[5] : 0.0f,
                out.size() > 6 ? out[6] : 0.0f,
                out.size() > 7 ? out[7] : 0.0f);

    if (bench.opts.dump_all) {
        std::printf("output_index,value\n");
        for (size_t i = 0; i < out.size(); ++i) {
            std::printf("%zu,%.9g\n", i, static_cast<double>(out[i]));
        }
    }

    ggml_backend_sched_free(bench.sched);
    ggml_backend_free(bench.backend);
    ggml_backend_free(bench.cpu_backend);
    return 0;
}
