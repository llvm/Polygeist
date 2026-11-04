// top level kernel of neural network

#include <cmath>
#include <cstdint>
#include "hls_stream.h"
#include "hls_math.h"

// defining sizes of inputs / layers
#define IN_CH   256
#define IN_H    40
#define IN_W    40
#define OUT_CH  512
#define OUT_H   20
#define OUT_W   20
#define FC_IN   OUT_CH
#define FC_OUT  512

// typedefs and constants
typedef float DataType;
const DataType EPSILON = 1e-5f;

// fast inverse square root implementation
inline DataType fast_rsqrt(DataType x) {
#pragma HLS INLINE
    union { DataType f; uint32_t i; } conv;
    conv.f = x;
    conv.i = 0x5f3759df - (conv.i >> 1);
    DataType y = conv.f;
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// 3x3 convolution
static void conv3x3_first_fast(
    const DataType in[IN_CH][IN_H][IN_W],
    DataType out[OUT_CH][OUT_H][OUT_W],
    const DataType w[OUT_CH][IN_CH][3][3],
    const DataType gamma[OUT_CH],
    const DataType beta[OUT_CH],
    const DataType mean[OUT_CH],
    const DataType var[OUT_CH],
    int stride
) {
    // break up padded for pipelining
    DataType padded[IN_CH][IN_H + 2][IN_W + 2];
    #pragma HLS ARRAY_PARTITION variable=padded cyclic factor=2 dim=1

    // quickly clear padded
    for (int c = 0; c < IN_CH; ++c) {
        for (int i = 0; i < IN_H + 2; ++i) {
            for (int j = 0; j < IN_W + 2; ++j) {
            #pragma HLS PIPELINE II=1
                padded[c][i][j] = 0.0f;
            }
        }
    }

    // pad inputs
    for (int c = 0; c < IN_CH; ++c) {
        for (int i = 0; i < IN_H; ++i) {
            for (int j = 0; j < IN_W; ++j) {
            #pragma HLS PIPELINE II=1
                padded[c][i + 1][j + 1] = in[c][i][j];
            }
        }
    }

    // break up acc for pipelining
    DataType acc[OUT_H][OUT_W];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    // 3x3 convolution, batchnorm, and relu
    for (int oc = 0; oc < OUT_CH; ++oc) {
        DataType inv_std = fast_rsqrt(var[oc] + EPSILON);
        DataType g = gamma[oc] * inv_std;
        DataType b = beta[oc] - gamma[oc] * mean[oc] * inv_std;

        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                acc[h][w_] = 0.0f;
            }
        }

        for (int ic = 0; ic < IN_CH; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw_ = 0; kw_ < 3; ++kw_) {
                #pragma HLS PIPELINE II=1
                    const DataType (*kernel_slice)[3] = w[oc][ic];
                    DataType kv = kernel_slice[kh][kw_];
                    int ih_base = kh;
                    int iw_base = kw_;

                    for (int oh = 0; oh < OUT_H; ++oh) {
                        int ih = oh * stride + ih_base;
                        for (int ow = 0; ow < OUT_W; ++ow) {
                            int iw = ow * stride + iw_base;
                            acc[oh][ow] += kv * padded[ic][ih][iw];
                        }
                    }
                }
            }
        }

        // relu
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                DataType y = g * acc[h][w_] + b;
                out[oc][h][w_] = (y > 0.0f) ? y : 0.0f;
            }
        }
    }
}

// split into two functions due to issues during synthsis
static void conv3x3_512_fast(
    const DataType in[OUT_CH][OUT_H][OUT_W],
    DataType out[OUT_CH][OUT_H][OUT_W],
    const DataType w[OUT_CH][OUT_CH][3][3],
    const DataType gamma[OUT_CH],
    const DataType beta[OUT_CH],
    const DataType mean[OUT_CH],
    const DataType var[OUT_CH],
    int stride
) {
    // padded split for pipeling
    DataType padded[OUT_CH][OUT_H + 2][OUT_W + 2];
    #pragma HLS ARRAY_PARTITION variable=padded cyclic factor=2 dim=1

    // quickly clear padded
    for (int c = 0; c < OUT_CH; ++c) {
        for (int i = 0; i < OUT_H + 2; ++i) {
            for (int j = 0; j < OUT_W + 2; ++j) {
            #pragma HLS PIPELINE II=1
                padded[c][i][j] = 0.0f;
            }
        }
    }

    // pad inputs
    for (int c = 0; c < OUT_CH; ++c) {
        for (int i = 0; i < OUT_H; ++i) {
            for (int j = 0; j < IN_W; ++j) {
            #pragma HLS PIPELINE II=1
                padded[c][i + 1][j + 1] = in[c][i][j];
            }
        }
    }

    // partitioning for pipelining
    DataType acc[OUT_H][OUT_W];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    // 3x3 convoluion + bn
    for (int oc = 0; oc < OUT_CH; ++oc) {
        DataType inv_std = fast_rsqrt(var[oc] + EPSILON);
        DataType g = gamma[oc] * inv_std;
        DataType b = beta[oc] - gamma[oc] * mean[oc] * inv_std;

        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                acc[h][w_] = 0.0f;
            }
        }

        for (int ic = 0; ic < OUT_CH; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw_ = 0; kw_ < 3; ++kw_) {
                #pragma HLS PIPELINE II=1
                    const DataType (*kernel_slice)[3] = w[oc][ic];
                    DataType kv = kernel_slice[kh][kw_];
                    int ih_base = kh;
                    int iw_base = kw_;

                    for (int oh = 0; oh < OUT_H; ++oh) {
                        int ih = oh * stride + ih_base;
                        for (int ow = 0; ow < OUT_W; ++ow) {
                            int iw = ow * stride + iw_base;
                            acc[oh][ow] += kv * padded[ic][ih][iw];
                        }
                    }
                }
            }
        }

        // relu
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                out[oc][h][w_] = g * acc[h][w_] + b;
            }
        }
    }
}

// 1x1 downsample / pooling
static void conv1x1_downsample(
    const DataType in[IN_CH][IN_H][IN_W],
    DataType out[OUT_CH][OUT_H][OUT_W],
    const DataType w_global[OUT_CH][IN_CH][1][1],
    int stride
) {
    // store into bram to save resources
    static DataType w_buffer[OUT_CH][IN_CH];
    #pragma HLS BIND_STORAGE variable=w_buffer type=RAM_T2P impl=BRAM

    // partitioning weights for parallelization
    const int WEIGHT_PARTITION_FACTOR = 32;
    #pragma HLS ARRAY_PARTITION variable=w_buffer cyclic factor=WEIGHT_PARTITION_FACTOR dim=2

    // burst read weights  
    for (int oc = 0; oc < OUT_CH; ++oc) {
        for (int ic = 0; ic < IN_CH; ++ic) {
            #pragma HLS PIPELINE II=1
            w_buffer[oc][ic] = w_global[oc][ic][0][0];
        }
    }

    // 
    for (int h = 0; h < OUT_H; ++h) {
        for (int w_ = 0; w_ < OUT_W; ++w_) {
            int ih = h * stride;
            int iw = w_ * stride;
            for (int oc = 0; oc < OUT_CH; ++oc) {
                #pragma HLS PIPELINE II=1
                DataType sum = 0.0f;
                for (int ic = 0; ic < IN_CH; ++ic) {
                    sum += w_buffer[oc][ic] * in[ic][ih][iw];
                }
                out[oc][h][w_] = sum;
            }
        }
    }
}

// batch normalization
static void batch_norm_512(
    DataType t[OUT_CH][OUT_H][OUT_W],
    const DataType gamma[OUT_CH],
    const DataType beta[OUT_CH],
    const DataType mean[OUT_CH],
    const DataType var[OUT_CH]
) {
    for (int c = 0; c < OUT_CH; ++c) {
        DataType inv_std = fast_rsqrt(var[c] + EPSILON);
        DataType g = gamma[c] * inv_std;
        DataType b = beta[c] - gamma[c] * mean[c] * inv_std;
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                t[c][h][w_] = g * t[c][h][w_] + b;
            }
        }
    }
}

// relu function
static void relu_512_array(DataType t[OUT_CH][OUT_H][OUT_W]) {
    for (int c = 0; c < OUT_CH; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                if (t[c][h][w_] < 0.0f) {
                    t[c][h][w_] = 0.0f;
                }
            }
        }
    }
}

// basic block from pytorch
static void basic_block_first_stream(
    const DataType in[IN_CH][IN_H][IN_W],
    hls::stream<DataType>& out_stream,
    const DataType c1w[OUT_CH][IN_CH][3][3], const DataType c1g[OUT_CH],
    const DataType c1b[OUT_CH], const DataType c1m[OUT_CH], const DataType c1v[OUT_CH],
    const DataType c2w[OUT_CH][OUT_CH][3][3], const DataType c2g[OUT_CH],
    const DataType c2b[OUT_CH], const DataType c2m[OUT_CH], const DataType c2v[OUT_CH],
    const DataType dw[OUT_CH][IN_CH][1][1], const DataType dg[OUT_CH],
    const DataType db[OUT_CH], const DataType dm[OUT_CH], const DataType dv[OUT_CH]
) {
    static DataType f1[OUT_CH][OUT_H][OUT_W];
    static DataType f2[OUT_CH][OUT_H][OUT_W];
    static DataType ds[OUT_CH][OUT_H][OUT_W];

    conv3x3_first_fast(in, f1, c1w, c1g, c1b, c1m, c1v, 2);
    conv3x3_512_fast(f1, f2, c2w, c2g, c2b, c2m, c2v, 1);

    conv1x1_downsample(in, ds, dw, 2);
    batch_norm_512(ds, dg, db, dm, dv);

    for (int c = 0; c < OUT_CH; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                DataType conv_path_val = f2[c][h][w_];
                DataType ds_path_val = ds[c][h][w_];
                DataType sum = conv_path_val + ds_path_val;
                DataType result = (sum > 0.0f) ? sum : 0.0f;
                out_stream.write(result);
            }
        }
    }
}

// basic block second from pytorch
static void basic_block_second_stream(
    hls::stream<DataType>& in_stream,
    hls::stream<DataType>& out_stream,
    const DataType c1w[OUT_CH][OUT_CH][3][3], const DataType c1g[OUT_CH],
    const DataType c1b[OUT_CH], const DataType c1m[OUT_CH], const DataType c1v[OUT_CH],
    const DataType c2w[OUT_CH][OUT_CH][3][3], const DataType c2g[OUT_CH],
    const DataType c2b[OUT_CH], const DataType c2m[OUT_CH], const DataType c2v[OUT_CH]
) {
    static DataType f0[OUT_CH][OUT_H][OUT_W];
    static DataType f1[OUT_CH][OUT_H][OUT_W];
    static DataType f2[OUT_CH][OUT_H][OUT_W];

    // internal stream
    static hls::stream<DataType> residual_stream("residual_stream_internal");
    #pragma HLS STREAM variable=residual_stream depth=1024 type=fifo

    for (int c = 0; c < OUT_CH; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                DataType val = in_stream.read();
                f0[c][h][w_] = val;
                residual_stream.write(val);
            }
        }
    }

    conv3x3_512_fast(f0, f1, c1w, c1g, c1b, c1m, c1v, 1);
    relu_512_array(f1);
    conv3x3_512_fast(f1, f2, c2w, c2g, c2b, c2m, c2v, 1);

    for (int c = 0; c < OUT_CH; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                DataType conv_path_val = f2[c][h][w_];
                DataType residual_val = residual_stream.read();
                DataType sum = conv_path_val + residual_val;
                DataType result = (sum > 0.0f) ? sum : 0.0f;
                out_stream.write(result);
            }
        }
    }
}

// average pooling
static void avg_pool_stream(
    hls::stream<DataType>& in_stream,
    hls::stream<DataType>& out_stream
) {
    static DataType sum_buffer[OUT_CH];
    DataType invHW = 1.0f / (DataType)(OUT_H * OUT_W);

    for (int c = 0; c < OUT_CH; ++c) {
    #pragma HLS PIPELINE II=1
        sum_buffer[c] = 0.0f;
    }

    for (int c = 0; c < OUT_CH; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w_ = 0; w_ < OUT_W; ++w_) {
            #pragma HLS PIPELINE II=1
                sum_buffer[c] += in_stream.read();
            }
        }
    }

    for (int c = 0; c < OUT_CH; ++c) {
    #pragma HLS PIPELINE II=1
        out_stream.write(sum_buffer[c] * invHW);
    }
}

// fully connected layer to output
static void fully_connected_stream(
    hls::stream<DataType>& in_stream,
    hls::stream<DataType>& out_stream,
    const DataType w[FC_OUT][FC_IN],
    const DataType b[FC_OUT]
) {
    static DataType in_buffer[FC_IN];

    for (int i = 0; i < FC_IN; ++i) {
    #pragma HLS PIPELINE II=1
        in_buffer[i] = in_stream.read();
    }

    // basic matrix multiplication for fully connectd layer
    for (int o = 0; o < FC_OUT; ++o) {
        DataType sum = b[o];
        for (int i = 0; i < FC_IN; ++i) {
        #pragma HLS PIPELINE II=1
            sum += w[o][i] * in_buffer[i];
        }
        out_stream.write(sum);
    }
}

// tanh implementation
static void tanh_stream(
    hls::stream<DataType>& in_stream,
    DataType out[FC_OUT]
) {
    for (int i = 0; i < FC_OUT; ++i) {
    #pragma HLS PIPELINE II=1
        DataType logit = in_stream.read();
        out[i] = tanhf(logit);
    }
}

// main krnel with weights, inpus, and output memory addresses
extern "C" void last_layer_kernel(
    const DataType in[IN_CH][IN_H][IN_W],
    DataType out[FC_OUT],

    const DataType b1w1[OUT_CH][IN_CH][3][3], const DataType b1g1[OUT_CH],
    const DataType b1b1[OUT_CH], const DataType b1m1[OUT_CH], const DataType b1v1[OUT_CH],
    const DataType b1w2[OUT_CH][OUT_CH][3][3], const DataType b1g2[OUT_CH],
    const DataType b1b2[OUT_CH], const DataType b1m2[OUT_CH], const DataType b1v2[OUT_CH],
    const DataType b1dw[OUT_CH][IN_CH][1][1], const DataType b1dg[OUT_CH],
    const DataType b1db[OUT_CH], const DataType b1dm[OUT_CH], const DataType b1dv[OUT_CH],
    const DataType b2w1[OUT_CH][OUT_CH][3][3], const DataType b2g1[OUT_CH],
    const DataType b2b1[OUT_CH], const DataType b2m1[OUT_CH], const DataType b2v1[OUT_CH],
    const DataType b2w2[OUT_CH][OUT_CH][3][3], const DataType b2g2[OUT_CH],
    const DataType b2b2[OUT_CH], const DataType b2m2[OUT_CH], const DataType b2v2[OUT_CH],
    const DataType fcw[FC_OUT][FC_IN], const DataType fcb[FC_OUT]
) {
// hls interface axi ports to optimize reads
#pragma HLS INTERFACE m_axi port=in   offset=slave bundle=gmem0 depth=409600  latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=out  offset=slave bundle=gmem0 depth=512     latency=100 num_write_outstanding=32

#pragma HLS INTERFACE m_axi port=b1w1 offset=slave bundle=gmem1 depth=1179648 latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1g1 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1b1 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1m1 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1v1 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1w2 offset=slave bundle=gmem1 depth=2359296 latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1g2 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1b2 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1m2 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1v2 offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1dw offset=slave bundle=gmem1 depth=131072  latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1dg offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1db offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1dm offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b1dv offset=slave bundle=gmem1 depth=512     latency=100 num_read_outstanding=32

#pragma HLS INTERFACE m_axi port=b2w1 offset=slave bundle=gmem2 depth=2359296 latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2g1 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2b1 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2m1 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2v1 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2w2 offset=slave bundle=gmem2 depth=2359296 latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2g2 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2b2 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2m2 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=b2v2 offset=slave bundle=gmem2 depth=512     latency=100 num_read_outstanding=32

#pragma HLS INTERFACE m_axi port=fcw offset=slave bundle=gmem3 depth=262144  latency=100 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=fcb offset=slave bundle=gmem3 depth=512     latency=100 num_read_outstanding=32

#pragma HLS INTERFACE s_axilite port=return bundle=control

    // stream definition
    static hls::stream<DataType> bb1_stream("bb1_stream");
    #pragma HLS STREAM variable=bb1_stream depth=1024 type=fifo

    static hls::stream<DataType> bb2_stream("bb2_stream");
    #pragma HLS STREAM variable=bb2_stream depth=1024 type=fifo

    static hls::stream<DataType> pool_stream("pool_stream");
    #pragma HLS STREAM variable=pool_stream depth=512 type=fifo

    static hls::stream<DataType> logits_stream("logits_stream");
    #pragma HLS STREAM variable=logits_stream depth=512 type=fifo

    // dataflowd architecture
    #pragma HLS DATAFLOW

    // first basic block
    basic_block_first_stream(in, bb1_stream,
                             b1w1, b1g1, b1b1, b1m1, b1v1,
                             b1w2, b1g2, b1b2, b1m2, b1v2,
                             b1dw, b1dg, b1db, b1dm, b1dv);
    // second basic block
    basic_block_second_stream(bb1_stream, bb2_stream,
                              b2w1, b2g1, b2b1, b2m1, b2v1,
                              b2w2, b2g2, b2b2, b2m2, b2v2);
    
    // average pool
    avg_pool_stream(bb2_stream, pool_stream);
    
    // fully connected layer
    fully_connected_stream(pool_stream, logits_stream, fcw, fcb);
    
    // output stream
    tanh_stream(logits_stream, out);

}