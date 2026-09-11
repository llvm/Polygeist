// Microbenchmark for the Llama-style suffix we currently raise:
//
//   hidden = rmsnorm(x) * weight
//   logits = W * hidden
//   probs  = softmax(logits)
//
// This intentionally mirrors third_party/cnn-extracted/llama2_forward_bench.c
// rather than a full llama.cpp token evaluation. Use it to compare the same
// suffix shape against ggml/CUDA.

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

namespace {

struct Options {
    int n = 2048;
    int h = 32000;
    int warmup = 5;
    int iters = 30;
    std::string stage = "suffix";
    bool identity_w = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--n N] [--h H] [--warmup W] [--iters I] "
                 "[--stage suffix|logits|hidden|norm|wcopy] [--identity-w]\n",
                 argv0);
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
        value > 2147483647L) {
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
        if (arg == "--n") {
            target = &opts.n;
        } else if (arg == "--h") {
            target = &opts.h;
        } else if (arg == "--warmup") {
            target = &opts.warmup;
        } else if (arg == "--iters") {
            target = &opts.iters;
        } else if (arg == "--stage") {
            if (++i >= argc) {
                usage(argv[0]);
                std::exit(2);
            }
            opts.stage = argv[i];
            if (opts.stage != "suffix" && opts.stage != "logits" &&
                opts.stage != "hidden" && opts.stage != "norm" &&
                opts.stage != "wcopy") {
                usage(argv[0]);
                std::exit(2);
            }
            continue;
        } else if (arg == "--identity-w") {
            opts.identity_w = true;
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
    return opts;
}

static void init_inputs(int n, int h, bool identity_w, std::vector<float> & x,
                        std::vector<float> & weight,
                        std::vector<float> & w) {
    x.resize(n);
    weight.resize(n);
    w.resize(static_cast<size_t>(h) * static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>((i % 31) - 15) * 0.0625f;
        weight[i] = 0.75f + static_cast<float>((i % 17) + 1) * 0.015625f;
    }

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < n; ++col) {
            if (identity_w) {
                w[static_cast<size_t>(row) * n + col] =
                    row == col ? 1.0f : 0.0f;
            } else {
                w[static_cast<size_t>(row) * n + col] =
                    static_cast<float>(((row * 7 + col * 11) % 29) - 14) *
                    0.0078125f;
            }
        }
    }
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

struct Bench {
    Options opts;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_buf;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * w = nullptr;
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

static void build_graph(Bench & bench) {
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
    bench.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bench.opts.n);
    bench.weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bench.opts.n);
    bench.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bench.opts.n, bench.opts.h);

    ggml_tensor * norm = ggml_rms_norm(ctx, bench.x, 1.0e-5f);
    ggml_tensor * norm_for_mul = ggml_cont(ctx, norm);
    ggml_tensor * hidden = ggml_mul(ctx, norm_for_mul, bench.weight);
    ggml_tensor * hidden_mat = ggml_reshape_2d(ctx, hidden, bench.opts.n, 1);
    ggml_tensor * logits_2d = ggml_mul_mat(ctx, hidden_mat, bench.w);
    ggml_tensor * logits_1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bench.opts.h);
    ggml_tensor * logits = ggml_cpy(ctx, logits_2d, logits_1d);
    if (bench.opts.stage == "wcopy") {
        bench.out = ggml_dup(ctx, bench.w);
    } else if (bench.opts.stage == "norm") {
        bench.out = norm;
    } else if (bench.opts.stage == "hidden") {
        bench.out = hidden;
    } else if (bench.opts.stage == "logits") {
        bench.out = logits_2d;
    } else {
        bench.out = ggml_soft_max(ctx, logits);
    }

    ggml_build_forward_expand(bench.graph, bench.out);
    ggml_free(ctx);
}

static void load_inputs(Bench & bench, const std::vector<float> & x,
                        const std::vector<float> & weight,
                        const std::vector<float> & w) {
    ggml_backend_sched_reset(bench.sched);
    if (!ggml_backend_sched_alloc_graph(bench.sched, bench.graph)) {
        std::fprintf(stderr, "failed to allocate ggml graph\n");
        std::exit(1);
    }

    if (bench.opts.stage != "wcopy") {
        ggml_backend_tensor_set(bench.x, x.data(), 0, ggml_nbytes(bench.x));
    }
    if (bench.opts.stage != "norm" && bench.opts.stage != "wcopy") {
        ggml_backend_tensor_set(bench.weight, weight.data(), 0,
                                ggml_nbytes(bench.weight));
    }
    if (bench.opts.stage != "hidden" && bench.opts.stage != "norm") {
        ggml_backend_tensor_set(bench.w, w.data(), 0, ggml_nbytes(bench.w));
    }
}

static double run_once(Bench & bench) {
    const int64_t t0 = ggml_time_us();
    const ggml_status status = ggml_backend_sched_graph_compute(
        bench.sched, bench.graph);
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

    std::vector<float> x;
    std::vector<float> weight;
    std::vector<float> w;
    init_inputs(bench.opts.n, bench.opts.h, bench.opts.identity_w, x, weight, w);

    init_backend(bench);
    build_graph(bench);
    load_inputs(bench, x, weight, w);

    std::fprintf(stderr, "backend=%s n=%d h=%d warmup=%d iters=%d stage=%s\n",
                 ggml_backend_name(bench.backend), bench.opts.n, bench.opts.h,
                 bench.opts.warmup, bench.opts.iters, bench.opts.stage.c_str());

    std::vector<double> times;
    for (int i = 0; i < bench.opts.warmup; ++i) {
        (void)run_once(bench);
    }

    times.reserve(bench.opts.iters);
    for (int i = 0; i < bench.opts.iters; ++i) {
        times.push_back(run_once(bench));
    }

    std::vector<float> out(static_cast<size_t>(ggml_nelements(bench.out)));
    ggml_backend_tensor_get(bench.out, out.data(), 0, ggml_nbytes(bench.out));

    double checksum = 0.0;
    for (float v : out) {
        checksum += static_cast<double>(v);
    }

    std::printf("bench,stage,backend,n,h,out_ne0,out_ne1,warmup,iters,avg_ms,median_ms,trimmed_ms,min_ms,max_ms,checksum,out0,out1,out2,out3\n");
    std::printf("ggml_suffix,%s,%s,%d,%d,%lld,%lld,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                bench.opts.stage.c_str(), ggml_backend_name(bench.backend),
                bench.opts.n, bench.opts.h,
                static_cast<long long>(bench.out->ne[0]),
                static_cast<long long>(bench.out->ne[1]), bench.opts.warmup,
                bench.opts.iters, average(times), median(times), trimmed_mean(times),
                *std::min_element(times.begin(), times.end()),
                *std::max_element(times.begin(), times.end()), checksum,
                out.size() > 0 ? out[0] : 0.0f,
                out.size() > 1 ? out[1] : 0.0f,
                out.size() > 2 ? out[2] : 0.0f,
                out.size() > 3 ? out[3] : 0.0f);

    ggml_backend_sched_free(bench.sched);
    ggml_backend_free(bench.backend);
    ggml_backend_free(bench.cpu_backend);
    return 0;
}
