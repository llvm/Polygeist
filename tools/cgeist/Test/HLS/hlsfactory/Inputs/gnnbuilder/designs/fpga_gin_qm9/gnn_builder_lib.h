#pragma once

///////////////////
// Array Helpers //
///////////////////

template <const int M, typename T>
void print_1d(T in[M]) {
    printf("[% 2.5f, % 2.5f, % 2.5f ... % 2.5f, % 2.5f, % 2.5f]\n",
           float(in[0]), (float)in[1], (float)in[2],
           float(in[M - 3]), (float)in[M - 2], (float)in[M - 1]);
}

template <const int M, const int N, typename T>
void print_2d(T in[M][N]) {
    print_1d<N, T>(in[0]);
    print_1d<N, T>(in[1]);
    print_1d<N, T>(in[2]);
    printf("[.............................................................]\n");
    printf("[.............................................................]\n");
    printf("[.............................................................]\n");
    print_1d<N, T>(in[M - 3]);
    print_1d<N, T>(in[M - 2]);
    print_1d<N, T>(in[M - 1]);
}

template <const int M, const int N, const int O, typename T>
void print_3d(T in[M][N][O]) {
    printf("[\n");
    print_2d<N, O, T>(in[0]);
    printf("                               .\n");
    printf("                               .\n");
    printf("                               .\n");
    print_2d<N, O, T>(in[M - 1]);
    printf("]\n");
}

template <const int M, typename T>
void copy_1d(T from[M], T to[M]) {
    for (int i = 0; i < M; i++) {
        to[i] = from[i];
    }
}

template <const int M, const int N, typename T>
void copy_2d(T from[M][N], T to[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            to[i][j] = from[i][j];
        }
    }
}

template <const int M, const int N, const int O, typename T>
void copy_3d(T from[M][N][O], T to[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                to[i][j][k] = from[i][j][k];
            }
        }
    }
}

template <const int M, typename T_in, typename T_out>
void cast_1d(T_in in[M], T_out out[M]) {
    for (int i = 0; i < M; i++) {
        out[i] = T_out(in[i]);
    }
}

template <const int M, const int N, typename T_in, typename T_out>
void cast_2d(T_in in[M][N], T_out out[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = T_out(in[i][j]);
        }
    }
}

template <const int M, const int N, const int O, typename T_in, typename T_out>
void cast_3d(T_in in[M][N][O], T_out out[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                out[i][j][k] = T_out(in[i][j][k]);
            }
        }
    }
}

template <const int M, typename T = float>
void load_data_1d(const char *fp, T arr[M]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_2d(const char *fp, T arr[M][N]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_3d(const char *fp, T arr[M][N][O]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N * O, f);
    fclose(f);
}

template <const int M, typename T = float>
void load_data_var_1d(const char *fp, T arr[M], int i) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_var_2d(const char *fp, T arr[M][N], int i, int j) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_var_3d(const char *fp, T arr[M][N][O], int i, int j, int k) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j * k, f);
    fclose(f);
}

template <const int M, typename T = float>
bool compare_data_1d(T arr1[M], T arr2[M], float eps) {
    for (int i = 0; i < M; i++) {
        if (std::abs(arr1[i] - arr2[i]) > eps) {
            return false;
        }
    }
    return true;
}

template <const int M, const int N, typename T = float>
bool compare_data_2d(T arr1[M][N], T arr2[M][N], float eps) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (std::abs(arr1[i][j] - arr2[i][j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

template <const int M, const int N, const int O, typename T = float>
bool compare_data_3d(T arr1[M][N][O], T arr2[M][N][O], float eps) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                if (std::abs(arr1[i][j][k] - arr2[i][j][k]) > eps) {
                    return false;
                }
            }
        }
    }
    return true;
}

// compute_mae_1d
template <const int M, typename T = float>
float compute_mae_1d(T arr1[M], T arr2[M]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        mae += std::abs(arr1[i] - arr2[i]);
    }
    return mae / M;
}

// compute_mae_2d
template <const int M, const int N, typename T = float>
float compute_mae_2d(T arr1[M][N], T arr2[M][N]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mae += std::abs(arr1[i][j] - arr2[i][j]);
        }
    }
    return mae / (M * N);
}

// compute_mae_3d
template <const int M, const int N, const int O, typename T = float>
float compute_mae_3d(T arr1[M][N][O], T arr2[M][N][O]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                mae += std::abs(arr1[i][j][k] - arr2[i][j][k]);
            }
        }
    }
    return mae / (M * N * O);
}

////////////////////
// Stream Helpers //
////////////////////

// template <typename T>
// T t_multiply(T arg) {
//     return arg;
// }

// template <typename T, typename... Args>
// T t_multiply(T arg, Args... args) {
//     return arg * t_multiply(args...);
// }

// template <int... _dim_sizes>
// struct array_shape {
//     const int n_dims = sizeof...(_dim_sizes);
//     const int dim_sizes[sizeof...(_dim_sizes)] = {_dim_sizes...};
//     const int N = t_multiply(_dim_sizes...);
// };

// template <typename T, typename array_shape>
// struct array_stream {
//     hls::stream<T> data;
//     using T_data = T;
//     const array_shape shape;
//     array_stream() {};
// };

///////////////////////////////
// Stream <-> Array Adapters //
///////////////////////////////

// template <const int M, typename T>
// void stream_to_array_1d(array_stream<T, array_shape<M>> &x_stream, T x_array[M]) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         x_array[i] = x_stream.data.read();
//     }
// }

// template <const int M, typename T>
// void array_to_stream_1d(T x_array[M], array_stream<T, array_shape<M>> &x_stream) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         x_stream.data.write(x_array[i]);
//     }
// }

// template <const int M, const int N, typename T, typename T_array = array_stream<T, array_shape<M, N>>>
// void stream_to_array_2d(T_array &x_stream, T x_array[M][N]) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             x_stream[i][j] = x_stream.data.read();
//         }
//     }
// }

// template <const int M, const int N, typename T, typename T_array = array_stream<T, array_shape<M, N>>>
// void array_to_stream_2d(T x_array[M][N], T_array &x_stream) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             x_stream.data.write(x_array[i][j]);
//         }
//     }
// }

// template <const int M, const int N, const int O, typename T, typename T_array = array_stream<T, array_shape<M, N, O>>>
// void stream_to_array_3d(T_array &x_stream, T x_array[M][N][O]) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             for (int k = 0; k < O; k++) {
//                 x_stream[i][j][k] = x_stream.data.read();
//             }
//         }
//     }
// }

// template <const int M, const int N, const int O, typename T, typename T_array = array_stream<T, array_shape<M, N, O>>>
// void array_to_stream_3d(T x_array[M][N][O], T_array &x_stream) {
// #pragma HLS INLINE off
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             for (int k = 0; k < O; k++) {
//                 x_stream.data.write(x_array[i][j][k]);
//             }
//         }
//     }
// }

//////////////////////////
// Activation functions //
//////////////////////////

template <typename T>
T activation_elu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T alpha = T(1.0);
    if (x > 0) {
        return x;
    } else {
        return alpha * (m_exp(x) - T(1.0));
    }
}

template <typename T>
T activation_hardtanh(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T min_val = T(-1.0);
    const T max_val = T(1.0);
    if (x < min_val) {
        return min_val;
    } else if (x > max_val) {
        return max_val;
    } else {
        return x;
    }
}

template <typename T>
T activation_leakyrelu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T negative_slope = T(0.1);

    if (x >= 0) {
        return x;
    } else {
        return x * negative_slope;
    }
}

template <typename T>
T activation_relu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

template <typename T>
T activation_gelu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T sqrt_2_recip = T(1.0) / m_sqrt(T(2.0));
    const T out = x * T(0.5) * (T(1.0) + m_erf(x * sqrt_2_recip));
    return out;
}

template <typename T>
T activation_gelu_approx_tanh(T x)
{   
#pragma HLS INLINE off
#pragma HLS PIPELINE

    const T GELU_APPROX_MIN = -8.31776613691702;
    const T GELU_TANH_COEFF_LINEAR = 0.7978845608028654;
    const T GELU_TANH_COEFF_CUBIC = 0.035677408136300125;
    const T GELU_APPROX_MAX = 8.31776613691702;

    // prevent overflow of tanh_arg by setting limits where the
    // datatype is too coarse to represent the tanh result anyway.
    if (x < GELU_APPROX_MIN) return T(0.0);
    if (x > GELU_APPROX_MAX) return x;

    const T tanh_arg = T(T(T(GELU_TANH_COEFF_CUBIC * x) * x) + GELU_TANH_COEFF_LINEAR) * x;
    const T tanh = m_tanh(tanh_arg);
    const T tanh_fixed = (m_signbit(tanh_arg) != m_signbit(tanh)) ? T(-tanh) : tanh;
    // #if  __FLOATING_POINT_MODEL__

#if __FLOATING_POINT_MODEL__
    T out = (x / T(2.0)) * T(T(1.0) + tanh_fixed);
#else
    T out = (x >> 1) * T(T(1.0) + tanh_fixed);
#endif

    return out;
}

template <typename T>
T activation_sigmoid(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return T(1.0) / (T(1.0) + m_exp(-x));
}

template <typename T>
T activation_silu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return x * (T(1.0) / (T(1.0) + m_exp(-x)));
}

template <typename T>
T activation_tanh(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
#if __FLOATING_POINT_MODEL__
    T out = m_tanh(x);
    return out;
#else
    T out = m_tanh(x);
    T out_fixed = (hls::signbit(x) != hls::signbit(out)) ? T(-out) : out;
    return out_fixed;
#endif
}

template <typename T>
T activation_softsign(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return x / (T(1.0) + m_abs(x));
}

template <typename T>
T activation_sin(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return m_sin(x);
}

template <typename T>
T activation_cos(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return m_cos(x);
}

template <typename T>
T activation_identity(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return x;
}

//////////////////////
// Apply Activation //
//////////////////////

// template <typename T_array, typename T_array::T_data (*activation_func)(typename T_array::T_data)>
// class Activation_Stream {
// public:
//     // template <typename T>
//     void forward(T_array &x_in, T_array &x_out){
//         #pragma HLS INLINE off
//         for (int i = 0; i < x_in.shape.N; i++) {
//             typename T_array::T_data x_in_buf = x_in.data.read();
//             typename T_array::T_data x_out_buf = activation_func(x_in_buf);
//             x_out.data.write(x_out_buf);
//         }
//     }
// };

//////////////////////
// Incremental mean //
//////////////////////

template <typename T>
struct mean_incremental_data {
    T sum = T(0.0);
    int count = 0;
    T mean = T(0.0);
    int finalized = 0;
};

template <typename T>
void mean_incremental_update(mean_incremental_data<T> &data, T x) {
// #pragma HLS INLINE off
    data.sum += x;
    data.count++;
    data.mean = data.sum / T(data.count);
}

template <typename T>
void mean_incremental_finalize(mean_incremental_data<T> &data) {
// #pragma HLS INLINE off
    data.finalized = 1;
}

//////////////////////////
// Incremental variance //
//////////////////////////

// use Welford's online algorithm

template <typename T>
struct variance_incremental_data {
    int count = 0;
    T mean = T(0.0);
    T m2 = T(0.0);
    T var = T(0.0);
    T std = T(0.0);
    int finalized = 0;
};

template <typename T>
void variance_incremental_update(variance_incremental_data<T> &data, T x) {
// #pragma HLS INLINE off
    data.count++;
    T delta = x - data.mean;
    data.mean += delta / T(data.count);
    data.m2 += delta * (x - data.mean);
}

template <typename T>
void variance_incremental_finalize(variance_incremental_data<T> &data) {
// #pragma HLS INLINE off
    data.var = data.m2 / T(data.count);
    data.std = m_sqrt(data.var + T(1e-5));
    data.finalized = 1;
}

/////////////////////
// Incremental sum //
/////////////////////

template <typename T>
struct sum_incremental_data {
    T sum = T(0.0);
    int finalized = 0;
};

template <typename T>
void sum_incremental_update(sum_incremental_data<T> &data, T x) {
// #pragma HLS INLINE off
    data.sum += x;
}

template <typename T>
void sum_incremental_finalize(sum_incremental_data<T> &data) {
// #pragma HLS INLINE off
    data.finalized = 1;
}

/////////////////////
// Incremental max //
/////////////////////

template <typename T>
struct max_incremental_data {
    T max = T(0.0);
    int one_sample_processed = 0;
    int finalized = 0;
};

template <typename T>
void max_incremental_update(max_incremental_data<T> &data, T x) {
// #pragma HLS INLINE off
    if (data.one_sample_processed == 0) {
        data.max = x;
        data.one_sample_processed = 1;
    } else {
        if (x > data.max) {
            data.max = x;
        }
    }
}

template <typename T>
void max_incremental_finalize(max_incremental_data<T> &data) {
// #pragma HLS INLINE off
    data.finalized = 1;
}

/////////////////////
// incremental min //
/////////////////////

template <typename T>
struct min_incremental_data {
    T min = T(0.0);
    int one_sample_processed = 0;
    int finalized = 0;
};

template <typename T>
void min_incremental_update(min_incremental_data<T> &data, T x) {
#pragma HLS INLINE off
    if (data.one_sample_processed == 0) {
        data.min = x;
        data.one_sample_processed = 1;
    } else {
        if (x < data.min) {
            data.min = x;
        }
    }
}

template <typename T>
void min_incremental_finalize(min_incremental_data<T> &data) {
#pragma HLS INLINE off
    data.finalized = 1;
}

//////////////////
// Linear Layer //
//////////////////

template <const int in_size, const int out_size,
          const int BLOCK_SIZE_IN_ = 1,
          const int BLOCK_SIZE_OUT_ = 1,
          typename T>
void linear(T input[in_size],
            T output[out_size],
            T weight[out_size][in_size],
            T bias[out_size]) {
#pragma HLS INLINE off

    static_assert(in_size % BLOCK_SIZE_IN_ == 0, "in_size must be divisible by BLOCK_SIZE_IN");
    static_assert(out_size % BLOCK_SIZE_OUT_ == 0, "out_size must be divisible by BLOCK_SIZE_OUT");

    const int BLOCK_SIZE_OUT = BLOCK_SIZE_OUT_;
    const int BLOCK_SIZE_IN = BLOCK_SIZE_IN_;

#pragma HLS array_partition variable = input cyclic factor = BLOCK_SIZE_IN dim = 1
#pragma HLS array_partition variable = output cyclic factor = BLOCK_SIZE_OUT dim = 1

#pragma HLS array_partition variable = weight cyclic factor = BLOCK_SIZE_OUT dim = 1
#pragma HLS array_partition variable = weight cyclic factor = BLOCK_SIZE_IN dim = 2

#pragma HLS array_partition variable = bias cyclic factor = BLOCK_SIZE_OUT dim = 1

    // block parallel linear layer
    // use temp sum
    F_TYPE temp_sum[BLOCK_SIZE_OUT];

#pragma HLS ARRAY_PARTITION variable = temp_sum complete

BLOCK_OUT:
    for (int i = 0; i < out_size; i += BLOCK_SIZE_OUT) {
    BLOCK_IN:
        for (int j = 0; j < in_size; j += BLOCK_SIZE_IN) {

#pragma HLS PIPELINE
        // zero temp sum
        TEMP_SUM_ZERO_LOOP:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {
                temp_sum[k] = 0;
            }

        // compute temp sum
        SUM_OUTER:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {
            SUM_INNER:
                for (int l = 0; l < BLOCK_SIZE_IN; l++) {
                    temp_sum[k] += weight[i + k][j + l] * input[j + l];
                }
            }

        // write temp sum to output
        // aslo write bias to output on first block itteration
        WRITE_LOOP:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {

                // check if first block itteration
                // if first block itteration, write bias
                if (j == 0) {
                    output[i + k] = bias[i + k];
                }

                output[i + k] += temp_sum[k];
            }
        }
    }
}

// linear buffered
template <const int in_size, const int out_size,
          const int BLOCK_SIZE_IN_ = 1,
          const int BLOCK_SIZE_OUT_ = 1,
          typename T>
void linear_buffered(T input[in_size],
                     T output[out_size],
                     T weight[out_size][in_size],
                     T bias[out_size]) {
#pragma HLS INLINE off

    static_assert(in_size % BLOCK_SIZE_IN_ == 0, "in_size must be divisible by BLOCK_SIZE_IN");
    static_assert(out_size % BLOCK_SIZE_OUT_ == 0, "out_size must be divisible by BLOCK_SIZE_OUT");

    const int BLOCK_SIZE_OUT = BLOCK_SIZE_OUT_;
    const int BLOCK_SIZE_IN = BLOCK_SIZE_IN_;

#pragma HLS array_partition variable = input cyclic factor = BLOCK_SIZE_IN dim = 1
#pragma HLS array_partition variable = output cyclic factor = BLOCK_SIZE_OUT dim = 1

#pragma HLS array_partition variable = weight cyclic factor = BLOCK_SIZE_OUT dim = 1
#pragma HLS array_partition variable = weight cyclic factor = BLOCK_SIZE_IN dim = 2

#pragma HLS array_partition variable = bias cyclic factor = BLOCK_SIZE_OUT dim = 1

    // block parallel linear layer
    // use temp sum
    F_TYPE temp_sum[BLOCK_SIZE_OUT];
    #pragma HLS ARRAY_PARTITION variable = temp_sum complete


    // output buffer
    F_TYPE output_buffer[out_size];
    #pragma HLS array_partition variable = output_buffer cyclic factor = BLOCK_SIZE_OUT dim = 1

    // zero output buffer

    BLOCK_OUT:
    for (int i = 0; i < out_size; i += BLOCK_SIZE_OUT) {
        BLOCK_IN:
        for (int j = 0; j < in_size; j += BLOCK_SIZE_IN) {

            #pragma HLS PIPELINE
            // zero temp sum
            TEMP_SUM_ZERO_LOOP:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {
                temp_sum[k] = 0;
            }

            // compute temp sum
            SUM_OUTER:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {
                SUM_INNER:
                for (int l = 0; l < BLOCK_SIZE_IN; l++) {
                    temp_sum[k] += weight[i + k][j + l] * input[j + l];
                }
            }

            // write temp sum to output
            // aslo write bias to output on first block itteration
            WRITE_LOOP:
            for (int k = 0; k < BLOCK_SIZE_OUT; k++) {

                // check if first block itteration
                // if first block itteration, write bias
                if (j == 0) {
                    output_buffer[i + k] = bias[i + k];
                }

                output_buffer[i + k] += temp_sum[k];
            }
        }
    }

    // write output buffer to output
    WRITE_OUTPUT_LOOP:
    for (int i = 0; i < out_size; i++) {
        output[i] = output_buffer[i];
    }
}

/////////////////////////
// simple linear layer //
/////////////////////////

template <const int in_size, const int out_size,
          typename T>
void linear_simple(T input[in_size],
                   T output[out_size],
                   T weight[out_size][in_size],
                   T bias[out_size]) {
#pragma HLS INLINE off

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            output[i] += weight[i][j] * input[j];
        }
        output[i] += bias[i];
    }
}

////////////////////////////////////
// graph representation functions //
////////////////////////////////////

template <int MAX_NODES, int MAX_EDGES, int NUM_NODES_GUESS = MAX_NODES, int NUM_EDGES_GUESS = MAX_EDGES>
void compute_degree_tables(int edge_list[MAX_EDGES][2],
                           int in_degree_table[MAX_NODES],
                           int out_degree_table[MAX_NODES],
                           int num_nodes,
                           int num_edges) {
#pragma HLS INLINE off

    //  edge list is a COO format
    //  edge_list[i][0] = source node
    //  edge_list[i][1] = destination node

    for (int i = 0; i < num_nodes; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS

        in_degree_table[i] = 0;
        out_degree_table[i] = 0;
    }

    for (int i = 0; i < num_edges; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_EDGES_GUESS
#pragma HLS PIPELINE off

        int source = edge_list[i][0];
        int dest = edge_list[i][1];

        in_degree_table[dest]++;
        out_degree_table[source]++;
    }
}

// compute_neighbor_table function
template <int MAX_NODES, int MAX_EDGES, int NUM_NODES_GUESS = MAX_NODES, int NUM_EDGES_GUESS = MAX_EDGES>
void compute_neighbor_tables(int edge_list[MAX_EDGES][2],
                             int in_degree_table[MAX_NODES],
                             int out_degree_table[MAX_NODES],
                             int neightbor_table_offsets[MAX_NODES],
                             int neighbor_table[MAX_EDGES],
                             int num_nodes,
                             int num_edges) {

#pragma HLS INLINE off

    // compute neighbor table offsets
    // cumulative sum of in_degree_table
    // also store values temp array for later
    int neightbor_table_offsets_temp[MAX_NODES];
    neightbor_table_offsets[0] = 0;
    neightbor_table_offsets_temp[0] = 0;
    for (int i = 1; i < num_nodes; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS
        int csum = neightbor_table_offsets[i - 1] + in_degree_table[i - 1];
        neightbor_table_offsets[i] = csum;
        neightbor_table_offsets_temp[i] = csum;
    }

    // compute neighbor table
    for (int i = 0; i < num_edges; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_EDGES_GUESS
#pragma HLS PIPELINE off
        int source = edge_list[i][0];
        int dest = edge_list[i][1];

        int offset = neightbor_table_offsets_temp[dest];
        neighbor_table[offset] = source;
        neightbor_table_offsets_temp[dest]++;
    }
}

template <int MAX_NODES, int MAX_EDGES, int NUM_NODES_GUESS = MAX_NODES, int NUM_EDGES_GUESS = MAX_EDGES, int DEGREE_GUESS = MAX_NODES>
void gather_node_neighbors(
    int node,
    int node_in_degree,
    int node_neighbors[MAX_NODES],
    int neightbor_table_offsets[MAX_NODES],
    int neighbor_table[MAX_EDGES]) {

// #pragma HLS INLINE off
#pragma HLS INLINE

    int node_offset = neightbor_table_offsets[node];
    for (int i = 0; i < node_in_degree; i++) {
#pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS
        node_neighbors[i] = neighbor_table[node_offset + i];
    }
}

///////////////
// gnn convs //
///////////////

template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE_IN,
          int EMB_SIZE_OUT,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES,
          int DEGREE_GUESS = MAX_NODES,
          int P_IN = 1,
          int P_OUT = 1>
void gcn_conv_agg(
    T current_node_embedding_in[EMB_SIZE_IN],
    int num_in_neighbors,
    int neighbors[MAX_NODES],
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    int in_degree_table[MAX_NODES],
    T agg_embedding[EMB_SIZE_IN]
){
#pragma HLS INLINE off

    sum_incremental_data<T> neighbor_emb_aggregation_sum[EMB_SIZE_IN];
    for (int neighbor = 0; neighbor < num_in_neighbors; neighbor++) {
        #pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS

        T neighbor_emb[EMB_SIZE_IN];

        int neighbor_id = neighbors[neighbor];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_emb[i] = node_embedding_table_in[neighbor_id][i];
        }

        int in_degree_neighbor = in_degree_table[neighbor_id];
        int in_degree_node = num_in_neighbors;

        T d_i_prime = T(1.0) + T(in_degree_node);
        T d_j_prime = T(1.0) + T(in_degree_neighbor);

        T degree_scaling_factor = m_recip(T(m_sqrt(d_i_prime * d_j_prime)));

        T neighbor_emb_transformed[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_emb_transformed[i] = neighbor_emb[i] * degree_scaling_factor;
        }

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            sum_incremental_update(neighbor_emb_aggregation_sum[i], neighbor_emb_transformed[i]);
        }
    }

    T d_i_prime = T(1.0) + num_in_neighbors;
    T degree_scaling_factor_self = m_recip(m_sqrt(d_i_prime * d_i_prime));

    T self_emb_transformed[EMB_SIZE_IN];
    for (int i = 0; i < EMB_SIZE_IN; i++) {
        self_emb_transformed[i] = current_node_embedding_in[i] * degree_scaling_factor_self;
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        sum_incremental_update(neighbor_emb_aggregation_sum[i], self_emb_transformed[i]);
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        sum_incremental_finalize(neighbor_emb_aggregation_sum[i]);
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        agg_embedding[i] = neighbor_emb_aggregation_sum[i].sum;
    }
}

template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE_IN,
          int EMB_SIZE_OUT,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES,
          int DEGREE_GUESS = MAX_NODES,
          int P_IN = 1,
          int P_OUT = 1>
void gcn_conv(
    int num_nodes,
    int num_edges,
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T node_embedding_table_out[MAX_NODES][EMB_SIZE_OUT],
    int edge_list[MAX_EDGES][2],
    int neightbor_table_offsets[MAX_NODES],
    int neighbor_table[MAX_EDGES],
    int in_degree_table[MAX_NODES],
    int out_degree_table[MAX_NODES],
    T apply_lin_weight[EMB_SIZE_OUT][EMB_SIZE_IN],
    T apply_lin_bias[EMB_SIZE_OUT]
) {

    #pragma HLS INLINE off

    for (int node = 0; node < num_nodes; node++) {
        #pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS

        #pragma HLS DATAFLOW
        #pragma HLS stable variable=edge_list
        #pragma HLS stable variable=neightbor_table_offsets
        #pragma HLS stable variable=neighbor_table
        #pragma HLS stable variable=in_degree_table
        #pragma HLS stable variable=out_degree_table

        T current_node_embedding_in[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            current_node_embedding_in[i] = node_embedding_table_in[node][i];
        }

        int num_in_neighbors = in_degree_table[node];
        int neighbors[MAX_NODES];

        gather_node_neighbors<
            MAX_NODES,
            MAX_EDGES,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS
        >(
            node,
            num_in_neighbors,
            neighbors,
            neightbor_table_offsets,
            neighbor_table
        );

        T agg_embedding[EMB_SIZE_IN];
        gcn_conv_agg<
            MAX_NODES,
            MAX_EDGES,
            EMB_SIZE_IN,
            EMB_SIZE_OUT,
            T,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS,
            P_IN,
            P_OUT
        >
        (
            current_node_embedding_in,
            num_in_neighbors,
            neighbors,
            node_embedding_table_in,
            in_degree_table,
            agg_embedding
        );

        // compute new node embedding
        T new_node_embedding[EMB_SIZE_OUT];
        linear<EMB_SIZE_IN, EMB_SIZE_OUT, P_IN, P_OUT, T>(agg_embedding, new_node_embedding, apply_lin_weight, apply_lin_bias);

        // node_embedding_table_out[node] = new_node_embedding;
        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            node_embedding_table_out[node][i] = new_node_embedding[i];
        }
    }
}

template <
    int MAX_NODES,
    int MAX_EDGES,
    int EMB_SIZE_IN,
    int EMB_SIZE_OUT,
    int HIDDEN_FEATURE_SIZE,
    typename T,
    int NUM_NODES_GUESS = MAX_NODES,
    int NUM_EDGES_GUESS = MAX_EDGES,
    int DEGREE_GUESS = MAX_NODES,
    int P_IN = 1,
    int P_OUT = 1
>
void gin_conv_agg(
    int num_in_neighbors,
    int neighbors[MAX_NODES],
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T agg_embedding[EMB_SIZE_IN]
){
    #pragma HLS INLINE off

    sum_incremental_data<T> neighbor_emb_aggregation_sum[EMB_SIZE_IN];
    for (int neighbor = 0; neighbor < num_in_neighbors; neighbor++) {
        #pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS

        T neighbor_emb[EMB_SIZE_IN];

        int neighbor_id = neighbors[neighbor];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_emb[i] = node_embedding_table_in[neighbor_id][i];
        }

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            sum_incremental_update(neighbor_emb_aggregation_sum[i], neighbor_emb[i]);
        }
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        sum_incremental_finalize(neighbor_emb_aggregation_sum[i]);
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        agg_embedding[i] = neighbor_emb_aggregation_sum[i].sum;
    }
}

// gin conv
template <
    int MAX_NODES,
    int MAX_EDGES,
    int EMB_SIZE_IN,
    int EMB_SIZE_OUT,
    int HIDDEN_FEATURE_SIZE,
    typename T,
    int NUM_NODES_GUESS = MAX_NODES,
    int NUM_EDGES_GUESS = MAX_EDGES,
    int DEGREE_GUESS = MAX_NODES,
    int P_IN = 1,
    int P_OUT = 1
>
void gin_conv(
    int num_nodes,
    int num_edges,
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T node_embedding_table_out[MAX_NODES][EMB_SIZE_OUT],
    int edge_list[MAX_EDGES][2],
    int neightbor_table_offsets[MAX_NODES],
    int neighbor_table[MAX_EDGES],
    int in_degree_table[MAX_NODES],
    int out_degree_table[MAX_NODES],
    T apply_mlp_0_weight[HIDDEN_FEATURE_SIZE][EMB_SIZE_IN],
    T apply_mlp_0_bias[HIDDEN_FEATURE_SIZE],
    T apply_mlp_1_weight[EMB_SIZE_OUT][HIDDEN_FEATURE_SIZE],
    T apply_mlp_1_bias[EMB_SIZE_OUT],
    T gin_eps
) {

    for (int node = 0; node < num_nodes; node++) {
        #pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS

        #pragma HLS DATAFLOW
        #pragma HLS stable variable=edge_list
        #pragma HLS stable variable=neightbor_table_offsets
        #pragma HLS stable variable=neighbor_table
        #pragma HLS stable variable=in_degree_table
        #pragma HLS stable variable=out_degree_table

        T current_node_embedding_in[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            current_node_embedding_in[i] = node_embedding_table_in[node][i];
        }

        int num_in_neighbors = in_degree_table[node];
        int neighbors[MAX_NODES];

        gather_node_neighbors<
            MAX_NODES,
            MAX_EDGES,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS>(
            node,
            num_in_neighbors,
            neighbors,
            neightbor_table_offsets,
            neighbor_table);

        T agg_embedding[EMB_SIZE_IN];
        gin_conv_agg<
            MAX_NODES,
            MAX_EDGES,
            EMB_SIZE_IN,
            EMB_SIZE_OUT,
            HIDDEN_FEATURE_SIZE,
            T,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS,
            P_IN,
            P_OUT
        >(
            num_in_neighbors,
            neighbors,
            node_embedding_table_in,
            agg_embedding
        );

        T self_emb_scaled[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            self_emb_scaled[i] = current_node_embedding_in[i] * (T(1) + gin_eps);
        }

        T agg_embedding_plus_scaled_self[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            agg_embedding_plus_scaled_self[i] = agg_embedding[i] + self_emb_scaled[i];
        }

        // compute new node embedding
        T new_node_embedding_hidden_emb[HIDDEN_FEATURE_SIZE];
        T new_node_embedding_hidden_emb_act[HIDDEN_FEATURE_SIZE];
        T new_node_embedding[EMB_SIZE_OUT];

        // linear<EMB_SIZE_IN, EMB_SIZE_OUT>(agg_embedding, new_node_embedding, apply_lin_weight, apply_lin_bias);
        linear<EMB_SIZE_IN, HIDDEN_FEATURE_SIZE, P_IN, P_IN, T>(agg_embedding_plus_scaled_self, new_node_embedding_hidden_emb, apply_mlp_0_weight, apply_mlp_0_bias);
        for (int i = 0; i < HIDDEN_FEATURE_SIZE; i++) {
            new_node_embedding_hidden_emb_act[i] = activation_relu<T>(new_node_embedding_hidden_emb[i]);
        }
        linear<HIDDEN_FEATURE_SIZE, EMB_SIZE_OUT, P_IN, P_OUT, T>(new_node_embedding_hidden_emb_act, new_node_embedding, apply_mlp_1_weight, apply_mlp_1_bias);

        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            node_embedding_table_out[node][i] = new_node_embedding[i];
        }
    }
}

template<
    int MAX_NODES,
    int MAX_EDGES,
    int EMB_SIZE_IN,
    int EMB_SIZE_OUT,
    int TRANSFORM_IN,
    int TRANSFORM_OUT,
    int APPLY_IN,
    int APPLY_OUT,
    typename T,
    int NUM_NODES_GUESS = MAX_NODES,
    int NUM_EDGES_GUESS = MAX_EDGES,
    int DEGREE_GUESS = MAX_NODES,
    int P_IN = 1,
    int P_OUT = 1
>
void pna_conv_agg(
    int num_in_neighbors,
    int neighbors[MAX_NODES],
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T current_node_embedding_in[EMB_SIZE_IN],
    T neighbor_emb_agg_max[EMB_SIZE_IN],
    T neighbor_emb_agg_min[EMB_SIZE_IN],
    T neighbor_emb_agg_mean[EMB_SIZE_IN],
    T neighbor_emb_agg_variance[EMB_SIZE_IN],
    T transfrom_lin_weight[TRANSFORM_OUT][TRANSFORM_IN],
    T transfrom_lin_bias[TRANSFORM_OUT]
){
    #pragma HLS INLINE off

    max_incremental_data<T> neighbor_emb_aggregation_max[EMB_SIZE_IN];
    min_incremental_data<T> neighbor_emb_aggregation_min[EMB_SIZE_IN];
    mean_incremental_data<T> neighbor_emb_aggregation_mean[EMB_SIZE_IN];
    variance_incremental_data<T> neighbor_emb_aggregation_variance[EMB_SIZE_IN];

    for (int neighbor = 0; neighbor < num_in_neighbors; neighbor++) {
        #pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS

        T neighbor_emb[EMB_SIZE_IN];

        int neighbor_id = neighbors[neighbor];

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_emb[i] = node_embedding_table_in[neighbor_id][i];
        }

        T neighbor_and_self_concat_emb[EMB_SIZE_IN * 2];

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_and_self_concat_emb[i] = current_node_embedding_in[i];
            neighbor_and_self_concat_emb[i + EMB_SIZE_IN] = neighbor_emb[i];
        }

        // tansform emb
        T transformed_emb[EMB_SIZE_IN] = {};
        linear<EMB_SIZE_IN * 2, EMB_SIZE_IN, P_IN, P_IN, T>(neighbor_and_self_concat_emb, transformed_emb, transfrom_lin_weight, transfrom_lin_bias);

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            max_incremental_update(neighbor_emb_aggregation_max[i], transformed_emb[i]);
            min_incremental_update(neighbor_emb_aggregation_min[i], transformed_emb[i]);
            mean_incremental_update(neighbor_emb_aggregation_mean[i], transformed_emb[i]);
            variance_incremental_update(neighbor_emb_aggregation_variance[i], transformed_emb[i]);
        }
    }

    // finalize
    for (int i = 0; i < EMB_SIZE_IN; i++) {
        max_incremental_finalize(neighbor_emb_aggregation_max[i]);
        min_incremental_finalize(neighbor_emb_aggregation_min[i]);
        mean_incremental_finalize(neighbor_emb_aggregation_mean[i]);
        variance_incremental_finalize(neighbor_emb_aggregation_variance[i]);
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        neighbor_emb_agg_max[i] = neighbor_emb_aggregation_max[i].max;
        neighbor_emb_agg_min[i] = neighbor_emb_aggregation_min[i].min;
        neighbor_emb_agg_mean[i] = neighbor_emb_aggregation_mean[i].mean;
        neighbor_emb_agg_variance[i] = neighbor_emb_aggregation_variance[i].std;
    }
}

template <
    int EMB_SIZE_IN,
    int CONCAT_SIZE,
    typename T
>
void pna_conv_concat(
    T current_node_embedding_in[EMB_SIZE_IN],
    T agg_max_identity_emb[EMB_SIZE_IN],
    T agg_min_identity_emb[EMB_SIZE_IN],
    T agg_mean_identity_emb[EMB_SIZE_IN],
    T agg_std_identity_emb[EMB_SIZE_IN],
    T agg_max_amplification[EMB_SIZE_IN],
    T agg_min_amplification[EMB_SIZE_IN],
    T agg_mean_amplification[EMB_SIZE_IN],
    T agg_std_amplification[EMB_SIZE_IN],
    T agg_max_attenuation[EMB_SIZE_IN],
    T agg_min_attenuation[EMB_SIZE_IN],
    T agg_mean_attenuation[EMB_SIZE_IN],
    T agg_std_attenuation[EMB_SIZE_IN],
    T pre_apply_emb[CONCAT_SIZE]
){
    #pragma HLS INLINE off
    for (int i = 0; i < EMB_SIZE_IN; i++) {
        pre_apply_emb[i + EMB_SIZE_IN*0] = current_node_embedding_in[i];

        pre_apply_emb[i + EMB_SIZE_IN*1] = agg_max_identity_emb[i];
        pre_apply_emb[i + EMB_SIZE_IN*2] = agg_min_identity_emb[i];
        pre_apply_emb[i + EMB_SIZE_IN*3] = agg_mean_identity_emb[i];
        pre_apply_emb[i + EMB_SIZE_IN*4] = agg_std_identity_emb[i];

        pre_apply_emb[i + EMB_SIZE_IN*5] = agg_max_amplification[i];
        pre_apply_emb[i + EMB_SIZE_IN*6] = agg_min_amplification[i];
        pre_apply_emb[i + EMB_SIZE_IN*7] = agg_mean_amplification[i];
        pre_apply_emb[i + EMB_SIZE_IN*8] = agg_std_amplification[i];

        pre_apply_emb[i + EMB_SIZE_IN*9] = agg_max_attenuation[i];
        pre_apply_emb[i + EMB_SIZE_IN*10] = agg_min_attenuation[i];
        pre_apply_emb[i + EMB_SIZE_IN*11] = agg_mean_attenuation[i];
        pre_apply_emb[i + EMB_SIZE_IN*12] = agg_std_attenuation[i];
    }
}

template<typename T>
void pna_conv_copy_num_in_neighbors(
    T &num_in_neighbors,
    T &num_in_neighbors_0,
    T &num_in_neighbors_1,
    T &num_in_neighbors_2
) {
    # pragma HLS INLINE off
    num_in_neighbors_0 = num_in_neighbors;
    num_in_neighbors_1 = num_in_neighbors;
    num_in_neighbors_2 = num_in_neighbors;
}

template <
    int MAX_NODES,
    int MAX_EDGES,
    int EMB_SIZE_IN,
    int EMB_SIZE_OUT,
    int TRANSFORM_IN,
    int TRANSFORM_OUT,
    int APPLY_IN,
    int APPLY_OUT,
    typename T,
    int NUM_NODES_GUESS = MAX_NODES,
    int NUM_EDGES_GUESS = MAX_EDGES,
    int DEGREE_GUESS = MAX_NODES,
    int P_IN = 1,
    int P_OUT = 1
>
void pna_conv(
    int num_nodes,
    int num_edges,
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T node_embedding_table_out[MAX_NODES][EMB_SIZE_OUT],
    int edge_list[MAX_EDGES][2],
    int neightbor_table_offsets[MAX_NODES],
    int neighbor_table[MAX_EDGES],
    int in_degree_table[MAX_NODES],
    int out_degree_table[MAX_NODES],
    T transfrom_lin_weight[TRANSFORM_OUT][TRANSFORM_IN],
    T transfrom_lin_bias[TRANSFORM_OUT],
    T apply_lin_weight[APPLY_OUT][APPLY_IN],
    T apply_lin_bias[APPLY_OUT],
    T final_lin_weight[EMB_SIZE_OUT][EMB_SIZE_OUT],
    T final_lin_bias[EMB_SIZE_OUT],
    T pna_avg_degree_log
) {
    #pragma HLS INLINE off

    for (int node = 0; node < num_nodes; node++) {
        #pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS

        #pragma HLS DATAFLOW
        #pragma HLS stable variable=edge_list
        #pragma HLS stable variable=neightbor_table_offsets
        #pragma HLS stable variable=neighbor_table
        #pragma HLS stable variable=in_degree_table
        #pragma HLS stable variable=out_degree_table

        T current_node_embedding_in[EMB_SIZE_IN];
        T current_node_embedding_in_for_agg[EMB_SIZE_IN];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            T x_in = node_embedding_table_in[node][i];
            current_node_embedding_in[i] = x_in;
            current_node_embedding_in_for_agg[i] = x_in;
        }

        int num_in_neighbors;
        int num_in_neighbors_0;
        int num_in_neighbors_1;
        int num_in_neighbors_2;

        num_in_neighbors = in_degree_table[node];
        // num_in_neighbors_0 = num_in_neighbors;
        // num_in_neighbors_1 = num_in_neighbors;
        // num_in_neighbors_2 = num_in_neighbors;
        // turn into a void function
        
        pna_conv_copy_num_in_neighbors(num_in_neighbors, num_in_neighbors_0, num_in_neighbors_1, num_in_neighbors_2);

        int neighbors[MAX_NODES];

        gather_node_neighbors<
            MAX_NODES,
            MAX_EDGES,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS>(
            node,
            num_in_neighbors_0,
            neighbors,
            neightbor_table_offsets,
            neighbor_table);

        int num_in_neighbors_clamped;
        if (num_in_neighbors_1 < 1) {
            num_in_neighbors_clamped = 1;
        } else {
            num_in_neighbors_clamped = num_in_neighbors_1;
        }

        T amplification_factor = m_log(T(num_in_neighbors_clamped + 1)) / pna_avg_degree_log;
        T attenuation_factor = pna_avg_degree_log / m_log(T(num_in_neighbors_clamped + 1));


        T agg_max_identity_emb[EMB_SIZE_IN];
        T agg_min_identity_emb[EMB_SIZE_IN];
        T agg_mean_identity_emb[EMB_SIZE_IN];
        T agg_std_identity_emb[EMB_SIZE_IN];



        pna_conv_agg<
            MAX_NODES,
            MAX_EDGES,
            EMB_SIZE_IN,
            EMB_SIZE_OUT,
            TRANSFORM_IN,
            TRANSFORM_OUT,
            APPLY_IN,
            APPLY_OUT,
            T,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS,
            P_IN,
            P_OUT
        >(
            num_in_neighbors_2,
            neighbors,
            node_embedding_table_in,
            current_node_embedding_in_for_agg,
            agg_max_identity_emb,
            agg_min_identity_emb,
            agg_mean_identity_emb,
            agg_std_identity_emb,
            transfrom_lin_weight,
            transfrom_lin_bias
        );

        // const int AGG_EMB_SIZE = EMB_SIZE_IN * 4;

        // F_TYPE agg_emb[AGG_EMB_SIZE];
        // #pragma HLS array_partition variable = agg_emb dim = 1 block factor = 4
        // for (int i = 0; i < EMB_SIZE_IN; i++) {
        //     agg_emb[i] = agg_max_identity_emb[i];
        //     agg_emb[i + EMB_SIZE_IN] = agg_min_identity_emb[i];
        //     agg_emb[i + EMB_SIZE_IN * 2] = agg_mean_identity_emb[i];
        //     agg_emb[i + EMB_SIZE_IN * 3] = agg_std_identity_emb[i];
        // }

        // T agg_amplification_emb[AGG_EMB_SIZE];
        // T agg_attenuation_emb[AGG_EMB_SIZE];

        // for (int i = 0; i < EMB_SIZE_IN * 4; i++) {
        //     agg_amplification_emb[i] = amplification_factor * agg_emb[i];
        //     agg_attenuation_emb[i] = attenuation_factor * agg_emb[i];
        // }

        T agg_max_identity_0[EMB_SIZE_IN];
        T agg_min_identity_0[EMB_SIZE_IN];
        T agg_mean_identity_0[EMB_SIZE_IN];
        T agg_std_identity_0[EMB_SIZE_IN];

        T agg_max_identity_1[EMB_SIZE_IN];
        T agg_min_identity_1[EMB_SIZE_IN];
        T agg_mean_identity_1[EMB_SIZE_IN];
        T agg_std_identity_1[EMB_SIZE_IN];

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            T x_agg_max = agg_max_identity_emb[i];
            T x_agg_min = agg_min_identity_emb[i];
            T x_agg_mean = agg_mean_identity_emb[i];
            T x_agg_std = agg_std_identity_emb[i];

            agg_max_identity_0[i] = x_agg_max;
            agg_min_identity_0[i] = x_agg_min;
            agg_mean_identity_0[i] = x_agg_mean;
            agg_std_identity_0[i] = x_agg_std;

            agg_max_identity_1[i] = x_agg_max;
            agg_min_identity_1[i] = x_agg_min;
            agg_mean_identity_1[i] = x_agg_mean;
            agg_std_identity_1[i] = x_agg_std;
        }


        T agg_max_amplification[EMB_SIZE_IN];
        T agg_min_amplification[EMB_SIZE_IN];
        T agg_mean_amplification[EMB_SIZE_IN];
        T agg_std_amplification[EMB_SIZE_IN];

        T agg_max_attenuation[EMB_SIZE_IN];
        T agg_min_attenuation[EMB_SIZE_IN];
        T agg_mean_attenuation[EMB_SIZE_IN];
        T agg_std_attenuation[EMB_SIZE_IN];

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            T x_agg_max = agg_max_identity_0[i];
            T x_agg_min = agg_min_identity_0[i];
            T x_agg_mean = agg_mean_identity_0[i];
            T x_agg_std = agg_std_identity_0[i];

            agg_max_amplification[i] = amplification_factor * x_agg_max;
            agg_min_amplification[i] = amplification_factor * x_agg_min;
            agg_mean_amplification[i] = amplification_factor * x_agg_mean;
            agg_std_amplification[i] = amplification_factor * x_agg_std;

            agg_max_attenuation[i] = attenuation_factor * x_agg_max;
            agg_min_attenuation[i] = attenuation_factor * x_agg_min;
            agg_mean_attenuation[i] = attenuation_factor * x_agg_mean;
            agg_std_attenuation[i] = attenuation_factor * x_agg_std;
        }

        const int concat_size = EMB_SIZE_IN * 3 * 4 + EMB_SIZE_IN;

        T pre_apply_emb[concat_size];
        // #pragma HLS array_partition variable = pre_apply_emb dim = 1 block factor = concat_size

        pna_conv_concat<
            EMB_SIZE_IN,
            concat_size,
            T
        >(
            current_node_embedding_in,
            agg_max_identity_1,
            agg_min_identity_1,
            agg_mean_identity_1,
            agg_std_identity_1,
            agg_max_amplification,
            agg_min_amplification,
            agg_mean_amplification,
            agg_std_amplification,
            agg_max_attenuation,
            agg_min_attenuation,
            agg_mean_attenuation,
            agg_std_attenuation,
            pre_apply_emb
        );

        // self, then all the aggregations, then all the amplification aggregations, then all the attenuation aggregations
        // for (int i = 0; i < EMB_SIZE_IN; i++) {
        //     pre_apply_emb[i] = current_node_embedding_in[i];
        // }

        // for (int i = 0; i < EMB_SIZE_IN; i++) {
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 0] = agg_max_identity_emb[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 1] = agg_min_identity_emb[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 2] = agg_mean_identity_emb[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 3] = agg_std_identity_emb[i];

        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 4] = agg_max_amplification[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 5] = agg_min_amplification[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 6] = agg_mean_amplification[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 7] = agg_std_amplification[i];

        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 8] = agg_max_attenuation[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 9] = agg_min_attenuation[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 10] = agg_mean_attenuation[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + EMB_SIZE_IN * 11] = agg_std_attenuation[i];
        // }

        // add the amplification and attenuation aggregations
        // for (int i = 0; i < AGG_EMB_SIZE; i++) {
        //     pre_apply_emb[i + EMB_SIZE_IN + AGG_EMB_SIZE * 0] = agg_emb[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + AGG_EMB_SIZE * 1] = agg_amplification_emb[i];
        //     pre_apply_emb[i + EMB_SIZE_IN + AGG_EMB_SIZE * 2] = agg_attenuation_emb[i];
        // }

        // apply
        T new_node_embedding_hidden[EMB_SIZE_OUT];
        T new_node_embedding[EMB_SIZE_OUT];

        linear<concat_size, EMB_SIZE_OUT, P_IN, P_OUT, T>(pre_apply_emb, new_node_embedding_hidden, apply_lin_weight, apply_lin_bias);
        linear<EMB_SIZE_OUT, EMB_SIZE_OUT, P_OUT, P_OUT, T>(new_node_embedding_hidden, new_node_embedding, final_lin_weight, final_lin_bias);

        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            node_embedding_table_out[node][i] = new_node_embedding[i];
        }
    }
}

// sage_conv

template <
    int MAX_NODES,
    int MAX_EDGES,
    int EMB_SIZE_IN,
    int EMB_SIZE_OUT,
    typename T,
    int NUM_NODES_GUESS = MAX_NODES,
    int NUM_EDGES_GUESS = MAX_EDGES,
    int DEGREE_GUESS = MAX_NODES,
    int P_IN = 1,
    int P_OUT = 1
>
void sage_conv_agg(
    int num_in_neighbors,
    int neighbors[MAX_NODES],
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T agg_embedding[EMB_SIZE_IN]
){
    #pragma HLS INLINE off

    mean_incremental_data<T> neighbor_emb_aggregation_mean[EMB_SIZE_IN];

    for (int neighbor = 0; neighbor < num_in_neighbors; neighbor++) {
        #pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS

        T neighbor_emb[EMB_SIZE_IN];

        int neighbor_id = neighbors[neighbor];
        for (int i = 0; i < EMB_SIZE_IN; i++) {
            neighbor_emb[i] = node_embedding_table_in[neighbor_id][i];
        }

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            mean_incremental_update(neighbor_emb_aggregation_mean[i], neighbor_emb[i]);
        }
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        mean_incremental_finalize(neighbor_emb_aggregation_mean[i]);
    }

    for (int i = 0; i < EMB_SIZE_IN; i++) {
        agg_embedding[i] = neighbor_emb_aggregation_mean[i].mean;
    }

}


template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE_IN,
          int EMB_SIZE_OUT,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES,
          int DEGREE_GUESS = MAX_NODES,
          int P_IN = 1,
          int P_OUT = 1>
void sage_conv(
    int num_nodes,
    int num_edges,
    T node_embedding_table_in[MAX_NODES][EMB_SIZE_IN],
    T node_embedding_table_out[MAX_NODES][EMB_SIZE_OUT],
    int edge_list[MAX_EDGES][2],
    int neightbor_table_offsets[MAX_NODES],
    int neighbor_table[MAX_EDGES],
    int in_degree_table[MAX_NODES],
    int out_degree_table[MAX_NODES],
    T neighbor_lin_weight[EMB_SIZE_OUT][EMB_SIZE_IN],
    T neighbor_lin_bias[EMB_SIZE_OUT],
    T self_lin_weight[EMB_SIZE_OUT][EMB_SIZE_IN]) {

    for (int node = 0; node < num_nodes; node++) {
        #pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS

        #pragma HLS DATAFLOW

        T current_node_embedding_in[EMB_SIZE_IN];

        for (int i = 0; i < EMB_SIZE_IN; i++) {
            current_node_embedding_in[i] = node_embedding_table_in[node][i];
        }

        int num_in_neighbors = in_degree_table[node];
        int neighbors[MAX_NODES];

        gather_node_neighbors<
            MAX_NODES,
            MAX_EDGES,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS>(
            node,
            num_in_neighbors,
            neighbors,
            neightbor_table_offsets,
            neighbor_table);

        // mean_incremental_data<T> neighbor_emb_aggregation_mean[EMB_SIZE_IN];

        // for (int neighbor = 0; neighbor < num_in_neighbors; neighbor++) {
        //     #pragma HLS loop_tripcount min = 0 max = DEGREE_GUESS

        //     T neighbor_emb[EMB_SIZE_IN];

        //     int neighbor_id = neighbors[neighbor];
        //     for (int i = 0; i < EMB_SIZE_IN; i++) {
        //         neighbor_emb[i] = node_embedding_table_in[neighbor_id][i];
        //     }

        //     for (int i = 0; i < EMB_SIZE_IN; i++) {
        //         mean_incremental_update(neighbor_emb_aggregation_mean[i], neighbor_emb[i]);
        //     }
        // }

        // for (int i = 0; i < EMB_SIZE_IN; i++) {
        //     mean_incremental_finalize(neighbor_emb_aggregation_mean[i]);
        // }

        // T agg_emb[EMB_SIZE_IN];
        // for (int i = 0; i < EMB_SIZE_IN; i++) {
        //     agg_emb[i] = neighbor_emb_aggregation_mean[i].mean;
        // }

        T agg_emb[EMB_SIZE_IN];
        sage_conv_agg<
            MAX_NODES,
            MAX_EDGES,
            EMB_SIZE_IN,
            EMB_SIZE_OUT,
            T,
            NUM_NODES_GUESS,
            NUM_EDGES_GUESS,
            DEGREE_GUESS,
            P_IN,
            P_OUT>(
            num_in_neighbors,
            neighbors,
            node_embedding_table_in,
            agg_emb);

        // transform the aggregation embedding
        T agg_emb_transformed[EMB_SIZE_OUT];
        linear<EMB_SIZE_IN, EMB_SIZE_OUT, P_IN, P_OUT, T>(agg_emb, agg_emb_transformed, neighbor_lin_weight, neighbor_lin_bias);

        // transform the current node embedding
        T self_emb_transformed[EMB_SIZE_OUT];
        T init = T(0.0);
        T self_lin_bias[EMB_SIZE_OUT];
        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            self_lin_bias[i] = init;
        }
        linear<EMB_SIZE_IN, EMB_SIZE_OUT, P_IN, P_OUT, T>(current_node_embedding_in, self_emb_transformed, self_lin_weight, self_lin_bias);

        // add the two embeddings
        T new_node_embedding[EMB_SIZE_OUT];
        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            new_node_embedding[i] = agg_emb_transformed[i] + self_emb_transformed[i];
        }
        
        // write to node_embedding_table_out
        for (int i = 0; i < EMB_SIZE_OUT; i++) {
            node_embedding_table_out[node][i] = new_node_embedding[i];
        }
    }
}

// TODO: GAT layer




////////////////////
// global pooling //
////////////////////

template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES>
void global_add_pool(int num_nodes,
                     int num_edges,
                     T node_embedding_table[MAX_NODES][EMB_SIZE],
                     T pooled_embedding[EMB_SIZE]) {
#pragma HLS INLINE off

    sum_incremental_data<T> sum_agg[EMB_SIZE];
    for (int i = 0; i < num_nodes; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS
        for (int j = 0; j < EMB_SIZE; j++) {
            sum_incremental_update(sum_agg[j], node_embedding_table[i][j]);
        }
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        sum_incremental_finalize(sum_agg[i]);
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        pooled_embedding[i] = sum_agg[i].sum;
    }
}

template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES>
void global_mean_pool(int num_nodes,
                      int num_edges,
                      T node_embedding_table[MAX_NODES][EMB_SIZE],
                      T pooled_embedding[EMB_SIZE]) {
#pragma HLS INLINE off

    mean_incremental_data<T> mean_agg[EMB_SIZE];
    for (int i = 0; i < num_nodes; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS
        for (int j = 0; j < EMB_SIZE; j++) {
            mean_incremental_update(mean_agg[j], node_embedding_table[i][j]);
        }
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        mean_incremental_finalize(mean_agg[i]);
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        pooled_embedding[i] = mean_agg[i].mean;
    }
}

template <int MAX_NODES,
          int MAX_EDGES,
          int EMB_SIZE,
          typename T,
          int NUM_NODES_GUESS = MAX_NODES,
          int NUM_EDGES_GUESS = MAX_EDGES>
void global_max_pool(int num_nodes,
                     int num_edges,
                     T node_embedding_table[MAX_NODES][EMB_SIZE],
                     T pooled_embedding[EMB_SIZE]) {
#pragma HLS INLINE off

    max_incremental_data<T> max_agg[EMB_SIZE];
    for (int i = 0; i < num_nodes; i++) {
#pragma HLS loop_tripcount min = 0 max = NUM_NODES_GUESS
        for (int j = 0; j < EMB_SIZE; j++) {
            max_incremental_update(max_agg[j], node_embedding_table[i][j]);
        }
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        max_incremental_finalize(max_agg[i]);
    }
    for (int i = 0; i < EMB_SIZE; i++) {
        pooled_embedding[i] = max_agg[i].max;
    }
}