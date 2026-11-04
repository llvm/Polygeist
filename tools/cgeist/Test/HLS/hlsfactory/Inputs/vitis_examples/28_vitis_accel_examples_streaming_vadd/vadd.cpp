/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*******************************************************************************
Description:

    This example uses the load/compute/store coding style which is generally
    the most efficient for implementing kernels using HLS. The load and store
    functions are responsible for moving data in and out of the kernel as
    efficiently as possible. The core functionality is decomposed across one
    of more compute functions. Whenever possible, the compute function should
    pass data through HLS streams and should contain a single set of nested loops.

    HLS stream objects are used to pass data between producer and consumer
    functions. Stream read and write operations have a blocking behavior which
    allows consumers and producers to synchronize with each other automatically.

    The dataflow pragma instructs the compiler to enable task-level pipelining.
    This is required for to load/compute/store functions to execute in a parallel
    and pipelined manner.

    The kernel operates on vectors of NUM_WORDS integers modeled using the hls::vector
    data type. This datatype provides intuitive support for parallelism and
    fits well the vector-add computation. The vector length is set to NUM_WORDS
    since NUM_WORDS integers amount to a total of 64 bytes, which is the maximum size of
    a kernel port. It is a good practice to match the compute bandwidth to the I/O
    bandwidth. Here the kernel loads, computes and stores NUM_WORDS integer values per
    clock cycle and is implemented as below:
                                       _____________
                                      |             |<----- Input Vector 1 from Global Memory
                                      |  load_input |       __
                                      |_____________|----->|  |
                                       _____________       |  | in1_stream
Input Vector 2 from Global Memory --->|             |      |__|
                               __     |  load_input |        |
                              |  |<---|_____________|        |
                   in2_stream |  |     _____________         |
                              |__|--->|             |<--------
                                      | compute_add |      __
                                      |_____________|---->|  |
                                       ______________     |  | out_stream
                                      |              |<---|__|
                                      | store_result |
                                      |______________|-----> Output result to Global Memory

*******************************************************************************/

// Includes
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"

#define MEMORY_DWIDTH 512
#define SIZEOF_WORD 4
#define NUM_WORDS ((MEMORY_DWIDTH) / (8 * SIZEOF_WORD))

#define DATA_SIZE 4096

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;

static void load_input(hls::vector<uint32_t, NUM_WORDS>* in,
                       hls::stream<hls::vector<uint32_t, NUM_WORDS> >& inStream,
                       int vSize) {
mem_rd:
    for (int i = 0; i < vSize; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        inStream << in[i];
    }
}

static void compute_add(hls::stream<hls::vector<uint32_t, NUM_WORDS> >& in1_stream,
                        hls::stream<hls::vector<uint32_t, NUM_WORDS> >& in2_stream,
                        hls::stream<hls::vector<uint32_t, NUM_WORDS> >& out_stream,
                        int vSize) {
// The kernel is operating with vector of NUM_WORDS integers. The + operator performs
// an element-wise add, resulting in NUM_WORDS parallel additions.
execute:
    for (int i = 0; i < vSize; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        out_stream << (in1_stream.read() + in2_stream.read());
    }
}

static void store_result(hls::vector<uint32_t, NUM_WORDS>* out,
                         hls::stream<hls::vector<uint32_t, NUM_WORDS> >& out_stream,
                         int vSize) {
mem_wr:
    for (int i = 0; i < vSize; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        out[i] = out_stream.read();
    }
}

extern "C" {

/*
    Vector Addition Kernel

    Arguments:
        in1  (input)  --> Input vector 1
        in2  (input)  --> Input vector 2
        out  (output) --> Output vector
        size (input)  --> Number of elements in vector
*/

void vadd(hls::vector<uint32_t, NUM_WORDS>* in1,
          hls::vector<uint32_t, NUM_WORDS>* in2,
          hls::vector<uint32_t, NUM_WORDS>* out,
          int size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0 depth = c_size
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1 depth = c_size
#pragma HLS INTERFACE m_axi port = out bundle = gmem0 depth = c_size

    static hls::stream<hls::vector<uint32_t, NUM_WORDS> > in1_stream("input_stream_1");
    static hls::stream<hls::vector<uint32_t, NUM_WORDS> > in2_stream("input_stream_2");
    static hls::stream<hls::vector<uint32_t, NUM_WORDS> > out_stream("output_stream");

    // Since NUM_WORDS values are processed
    // in parallel per loop iteration, the for loop only needs to iterate 'size / NUM_WORDS' times.
    assert(size % NUM_WORDS == 0);
    int vSize = size / NUM_WORDS;
#pragma HLS dataflow

    load_input(in1, in1_stream, vSize);
    load_input(in2, in2_stream, vSize);
    compute_add(in1_stream, in2_stream, out_stream, vSize);
    store_result(out, out_stream, vSize);
}
}
