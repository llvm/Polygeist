
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
  hls::stream<float> &v0,
  float v1[1][8][112][112]
) {	// L8
  loop0: for (int v2 = 0; v2 < 112; v2++) {	// L9
    loop1: for (int v3 = 0; v3 < 112; v3++) {	// L10
      loop2: for (int v4 = 0; v4 < 8; v4++) {	// L11
        #pragma HLS pipeline II=1
        float v5 = v0.read();	// L12
        v1[0][v4][v3][v2] = v5;	// L13
      }
    }
  }
  return ;	// L17
}

void node1(
  hls::stream<float> &v6,
  hls::stream<float> &v7,
  double v8,
  float v9,
  float v10
) {	// L19
  loop3: for (int v11 = 0; v11 < 112; v11++) {	// L20
    loop4: for (int v12 = 0; v12 < 112; v12++) {	// L21
      loop5: for (int v13 = 0; v13 < 8; v13++) {	// L22
        #pragma HLS pipeline II=1
        float v14 = v6.read();	// L23
        float v15 = v8;	// L24
        float v16 = v15 + v9;	// L25
        float v17 = 1.0 / sqrt(v16);	// L26
        float v18 = v14 * v17;	// L27
        float v19 = v18 + v10;	// L28
        bool v20 = v19 > v10;	// L29
        float v21 = v20 ? (float)v19 : (float)v10;	// L30
        v7.write(v21);	// L31
      }
    }
  }
  return ;	// L35
}

void node2(
  hls::stream<float> &v22,
  float v23[8][8][1][1],
  hls::stream<float> &v24,
  float v25
) {	// L37
  float v26[8];	// L38
  float v27;	// L39
  loop6: for (int v28 = 0; v28 < 112; v28++) {	// L40
    loop7: for (int v29 = 0; v29 < 112; v29++) {	// L41
      loop8: for (int v30 = 0; v30 < 8; v30++) {	// L42
        loop9: for (int v31 = 0; v31 < 8; v31++) {	// L43
          #pragma HLS pipeline II=1
          if (v31 == 0) {	// L44
            float v32 = v22.read();	// L45
            v27 = v32;	// L46
          }
          float v33 = v27;	// L48
          float v34 = v23[v31][v30][0][0];	// L49
          if (v30 == 0) {	// L50
            v26[v31] = v25;	// L51
          }
          float v35 = v26[v31];	// L53
          float v36 = v33 * v34;	// L54
          float v37 = v35 + v36;	// L55
          v26[v31] = v37;	// L56
          if ((v30 - 7) == 0) {	// L57
            float v38 = v26[v31];	// L58
            v24.write(v38);	// L59
          }
        }
      }
    }
  }
  return ;	// L65
}

void node3(
  float v39[8][3][3],
  hls::stream<float> &v40,
  hls::stream<float> &v41,
  float v42
) {	// L67
  float v43[8];	// L68
  float v44[8][114][114];	// L69
  loop10: for (int v45 = 0; v45 < 114; v45++) {	// L70
    loop11: for (int v46 = 0; v46 < 114; v46++) {	// L71
      loop12: for (int v47 = 0; v47 < 3; v47++) {	// L72
        loop13: for (int v48 = 0; v48 < 3; v48++) {	// L73
          loop14: for (int v49 = 0; v49 < 8; v49++) {	// L74
            #pragma HLS pipeline II=1
            if (v48 == 0 && v47 == 0) {	// L75
              float v50 = v40.read();	// L76
              v44[v49][v46][v45] = v50;	// L77
            }
            if ((v46 - 2) >= 0 && (v45 - 2) >= 0) {	// L79
              float v51 = v44[v49][((v48 + v46) - 2)][((v47 + v45) - 2)];	// L80
              float v52 = v39[v49][v48][v47];	// L81
              if (v48 == 0 && v47 == 0) {	// L82
                v43[v49] = v42;	// L83
              }
              float v53 = v43[v49];	// L85
              float v54 = v51 * v52;	// L86
              float v55 = v53 + v54;	// L87
              v43[v49] = v55;	// L88
              if ((v48 - 2) == 0 && (v47 - 2) == 0) {	// L89
                float v56 = v43[v49];	// L90
                v41.write(v56);	// L91
              }
            }
          }
        }
      }
    }
  }
  return ;	// L99
}

void node4(
  float v57[1][8][112][112],
  hls::stream<float> &v58,
  float v59
) {	// L101
  float v60[8][114][114];	// L102
  loop15: for (int v61 = 0; v61 < 114; v61++) {	// L103
    loop16: for (int v62 = 0; v62 < 114; v62++) {	// L104
      loop17: for (int v63 = 0; v63 < 8; v63++) {	// L105
        #pragma HLS pipeline II=1
        v60[v63][v62][v61] = v59;	// L106
      }
    }
  }
  loop18: for (int v64 = 0; v64 < 114; v64++) {	// L110
    loop19: for (int v65 = 0; v65 < 114; v65++) {	// L111
      loop20: for (int v66 = 0; v66 < 8; v66++) {	// L112
        #pragma HLS pipeline II=1
        if ((v65 - 1) >= 0 && (v64 - 1) >= 0 && ((-v65) + 112) >= 0 && ((-v64) + 112) >= 0) {	// L113
          float v67 = v57[0][v66][(v65 - 1)][(v64 - 1)];	// L114
          v60[v66][v65][v64] = v67;	// L115
        }
        float v68 = v60[v66][v65][v64];	// L117
        v58.write(v68);	// L118
      }
    }
  }
  return ;	// L122
}

void forward(
  float v69[1][8][112][112],
  float v70[8][8][1][1],
  float v71[8][3][3],
  float v72[1][8][112][112]
) {	// L124
	#pragma HLS DATAFLOW
  hls::stream<float> v73("fifo_0");
	#pragma HLS STREAM variable=v73 depth=100352	// L128
  hls::stream<float> v74("fifo_1");
	#pragma HLS STREAM variable=v74 depth=100352	// L129
  hls::stream<float> v75("fifo_2");
	#pragma HLS STREAM variable=v75 depth=100352	// L130
  hls::stream<float> v76("fifo_3");
	#pragma HLS STREAM variable=v76 depth=103968	// L131
  node4(v69, v76, 0.000000);	// L132
  node3(v71, v76, v75, 0.000000);	// L133
  node2(v75, v70, v74, 0.000000);	// L134
  node1(v74, v73, 0.000010, 1.000000, 0.000000);	// L135
  node0(v73, v72);	// L136
  return ;	// L137
}

