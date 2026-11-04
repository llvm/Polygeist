
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_half.h>
#include <cassert>
#include <hls_math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace std;

void node0(
  hls::stream<float> v0[10],
  float v1[410]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=10

  loop0: for (int v2 = 0; v2 < 41; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop1: for (int v3 = 0; v3 < 10; v3++) {
      float v4 = v0[v3].read();
      v1[(v3 + (v2 * 10))] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> v5[15],
  float v6[390]
) {
  #pragma HLS array_partition variable=v6 cyclic dim=1 factor=15

  loop2: for (int v7 = 0; v7 < 26; v7++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop3: for (int v8 = 0; v8 < 15; v8++) {
      float v9 = v5[v8].read();
      v6[(v8 + (v7 * 15))] = v9;
    }
  }
  return ;
}

void node2(
  float v10[410][390],
  float v11[390],
  hls::stream<float> v12[10],
  float v13
) {
  #pragma HLS array_partition variable=v10 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v10 cyclic dim=2 factor=15

  #pragma HLS array_partition variable=v11 cyclic dim=1 factor=15

  float v14[410];
  #pragma HLS array_partition variable=v14 cyclic dim=1 factor=10

  loop4: for (int v15 = 0; v15 < 26; v15++) {
    loop5: for (int v16 = 0; v16 < 41; v16++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v17 = 0; v17 < 15; v17++) {
        loop7: for (int v18 = 0; v18 < 10; v18++) {
          float v19 = v10[(v18 + (v16 * 10))][(v17 + (v15 * 15))];
          float v20 = v11[(v17 + (v15 * 15))];
          if ((v17 + (v15 * 15)) == 0) {
            v14[(v18 + (v16 * 10))] = v13;
          }
          float v21 = v14[(v18 + (v16 * 10))];
          float v22 = v19 * v20;
          float v23 = v21 + v22;
          v14[(v18 + (v16 * 10))] = v23;
          if (((v17 + (v15 * 15)) - 389) == 0) {
            float v24 = v14[(v18 + (v16 * 10))];
            v12[v18].write(v24);
          }
        }
      }
    }
  }
  return ;
}

void node3(
  float v25[410],
  float v26[410][390],
  hls::stream<float> v27[15],
  float v28
) {
  #pragma HLS array_partition variable=v25 cyclic dim=1 factor=10

  #pragma HLS array_partition variable=v26 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v26 cyclic dim=2 factor=15

  float v29[390];
  #pragma HLS array_partition variable=v29 cyclic dim=1 factor=15

  loop8: for (int v30 = 0; v30 < 41; v30++) {
    loop9: for (int v31 = 0; v31 < 26; v31++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop10: for (int v32 = 0; v32 < 10; v32++) {
        loop11: for (int v33 = 0; v33 < 15; v33++) {
          float v34 = v25[(v32 + (v30 * 10))];
          float v35 = v26[(v32 + (v30 * 10))][(v33 + (v31 * 15))];
          if ((v32 + (v30 * 10)) == 0) {
            v29[(v33 + (v31 * 15))] = v28;
          }
          float v36 = v29[(v33 + (v31 * 15))];
          float v37 = v34 * v35;
          float v38 = v36 + v37;
          v29[(v33 + (v31 * 15))] = v38;
          if (((v32 + (v30 * 10)) - 409) == 0) {
            float v39 = v29[(v33 + (v31 * 15))];
            v27[v33].write(v39);
          }
        }
      }
    }
  }
  return ;
}

void forward(
  float v40[410][390],
  float v41[410][390],
  float v42[410],
  float v43[390],
  float v44[390],
  float v45[410]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v40 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v40 cyclic dim=2 factor=15

  #pragma HLS array_partition variable=v41 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v41 cyclic dim=2 factor=15

  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=10

  #pragma HLS array_partition variable=v43 cyclic dim=1 factor=15

  #pragma HLS array_partition variable=v44 cyclic dim=1 factor=15

  #pragma HLS array_partition variable=v45 cyclic dim=1 factor=10

  hls::stream<float> v46[10];
	#pragma HLS STREAM variable=v46 depth=41
  hls::stream<float> v47[15];
	#pragma HLS STREAM variable=v47 depth=26
  node3(v42, v40, v47, 0.000000);
  node2(v41, v43, v46, 0.000000);
  node1(v47, v44);
  node0(v46, v45);
  return ;
}

