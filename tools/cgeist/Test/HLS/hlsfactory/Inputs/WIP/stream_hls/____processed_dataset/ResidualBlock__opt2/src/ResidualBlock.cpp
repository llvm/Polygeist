
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
  float v1[1][16][56][56]
) {	// L6
  loop0: for (int v2 = 0; v2 < 56; v2++) {	// L7
    loop1: for (int v3 = 0; v3 < 56; v3++) {	// L8
      loop2: for (int v4 = 0; v4 < 16; v4++) {	// L9
        #pragma HLS pipeline II=1
        float v5 = v0.read();	// L10
        v1[0][v4][v2][v3] = v5;	// L11
      }
    }
  }
  return ;	// L15
}

void node1(
  float v6[1][16][56][56],
  hls::stream<float> &v7,
  hls::stream<float> &v8,
  double v9,
  float v10,
  float v11
) {	// L17
  loop3: for (int v12 = 0; v12 < 56; v12++) {	// L18
    loop4: for (int v13 = 0; v13 < 56; v13++) {	// L19
      loop5: for (int v14 = 0; v14 < 16; v14++) {	// L20
        #pragma HLS pipeline II=1
        float v15 = v7.read();	// L21
        float v16 = v6[0][v14][v12][v13];	// L22
        float v17 = v9;	// L23
        float v18 = v17 + v11;	// L24
        float v19 = 1.0 / sqrt(v18);	// L25
        float v20 = v15 * v19;	// L26
        float v21 = v20 + v10;	// L27
        float v22 = v21 + v16;	// L28
        bool v23 = v22 > v10;	// L29
        float v24 = v23 ? (float)v22 : (float)v10;	// L30
        v8.write(v24);	// L31
      }
    }
  }
  return ;	// L35
}

void node2(
  hls::stream<float> &v25,
  float v26[16][16][3][3],
  hls::stream<float> &v27,
  float v28
) {	// L37
  float v29[16];	// L38
  float v30[16][58][58];	// L39
  loop6: for (int v31 = 0; v31 < 58; v31++) {	// L40
    loop7: for (int v32 = 0; v32 < 58; v32++) {	// L41
      loop8: for (int v33 = 0; v33 < 16; v33++) {	// L42
        loop9: for (int v34 = 0; v34 < 3; v34++) {	// L43
          loop10: for (int v35 = 0; v35 < 3; v35++) {	// L44
            loop11: for (int v36 = 0; v36 < 16; v36++) {	// L45
              #pragma HLS pipeline II=1
              if (v36 == 0 && v34 == 0 && v35 == 0) {	// L46
                float v37 = v25.read();	// L47
                v30[v33][v31][v32] = v37;	// L48
              }
              if ((v31 - 2) >= 0 && (v32 - 2) >= 0) {	// L50
                float v38 = v30[v33][((v34 + v31) - 2)][((v35 + v32) - 2)];	// L51
                float v39 = v26[v36][v33][v34][v35];	// L52
                if (v33 == 0 && v34 == 0 && v35 == 0) {	// L53
                  v29[v36] = v28;	// L54
                }
                float v40 = v29[v36];	// L56
                float v41 = v38 * v39;	// L57
                float v42 = v40 + v41;	// L58
                v29[v36] = v42;	// L59
                if ((v33 - 15) == 0 && (v34 - 2) == 0 && (v35 - 2) == 0) {	// L60
                  float v43 = v29[v36];	// L61
                  v27.write(v43);	// L62
                }
              }
            }
          }
        }
      }
    }
  }
  return ;	// L71
}

void node3(
  hls::stream<float> &v44,
  hls::stream<float> &v45,
  float v46
) {	// L73
  float v47[16][58][58];	// L74
  loop12: for (int v48 = 0; v48 < 58; v48++) {	// L75
    loop13: for (int v49 = 0; v49 < 58; v49++) {	// L76
      loop14: for (int v50 = 0; v50 < 16; v50++) {	// L77
        #pragma HLS pipeline II=1
        v47[v50][v48][v49] = v46;	// L78
      }
    }
  }
  loop15: for (int v51 = 0; v51 < 58; v51++) {	// L82
    loop16: for (int v52 = 0; v52 < 58; v52++) {	// L83
      loop17: for (int v53 = 0; v53 < 16; v53++) {	// L84
        #pragma HLS pipeline II=1
        if ((v51 - 1) >= 0 && (v52 - 1) >= 0 && ((-v51) + 56) >= 0 && ((-v52) + 56) >= 0) {	// L85
          float v54 = v44.read();	// L86
          v47[v53][v51][v52] = v54;	// L87
        }
        float v55 = v47[v53][v51][v52];	// L89
        v45.write(v55);	// L90
      }
    }
  }
  return ;	// L94
}

void node4(
  hls::stream<float> &v56,
  hls::stream<float> &v57,
  double v58,
  float v59,
  float v60
) {	// L96
  loop18: for (int v61 = 0; v61 < 56; v61++) {	// L97
    loop19: for (int v62 = 0; v62 < 56; v62++) {	// L98
      loop20: for (int v63 = 0; v63 < 16; v63++) {	// L99
        #pragma HLS pipeline II=1
        float v64 = v56.read();	// L100
        float v65 = v58;	// L101
        float v66 = v65 + v59;	// L102
        float v67 = 1.0 / sqrt(v66);	// L103
        float v68 = v64 * v67;	// L104
        float v69 = v68 + v60;	// L105
        bool v70 = v69 > v60;	// L106
        float v71 = v70 ? (float)v69 : (float)v60;	// L107
        v57.write(v71);	// L108
      }
    }
  }
  return ;	// L112
}

void node5(
  float v72[16][16][3][3],
  hls::stream<float> &v73,
  hls::stream<float> &v74,
  float v75
) {	// L114
  float v76[16];	// L115
  float v77[16][58][58];	// L116
  loop21: for (int v78 = 0; v78 < 58; v78++) {	// L117
    loop22: for (int v79 = 0; v79 < 58; v79++) {	// L118
      loop23: for (int v80 = 0; v80 < 3; v80++) {	// L119
        loop24: for (int v81 = 0; v81 < 3; v81++) {	// L120
          loop25: for (int v82 = 0; v82 < 16; v82++) {	// L121
            loop26: for (int v83 = 0; v83 < 16; v83++) {	// L122
              #pragma HLS pipeline II=1
              if (v83 == 0 && v80 == 0 && v81 == 0) {	// L123
                float v84 = v73.read();	// L124
                v77[v82][v78][v79] = v84;	// L125
              }
              if ((v78 - 2) >= 0 && (v79 - 2) >= 0) {	// L127
                float v85 = v77[v82][((v80 + v78) - 2)][((v81 + v79) - 2)];	// L128
                float v86 = v72[v83][v82][v80][v81];	// L129
                if (v82 == 0 && v80 == 0 && v81 == 0) {	// L130
                  v76[v83] = v75;	// L131
                }
                float v87 = v76[v83];	// L133
                float v88 = v85 * v86;	// L134
                float v89 = v87 + v88;	// L135
                v76[v83] = v89;	// L136
                if ((v82 - 15) == 0 && (v80 - 2) == 0 && (v81 - 2) == 0) {	// L137
                  float v90 = v76[v83];	// L138
                  v74.write(v90);	// L139
                }
              }
            }
          }
        }
      }
    }
  }
  return ;	// L148
}

void node6(
  float v91[1][16][56][56],
  hls::stream<float> &v92,
  float v93
) {	// L150
  float v94[16][58][58];	// L151
  loop27: for (int v95 = 0; v95 < 58; v95++) {	// L152
    loop28: for (int v96 = 0; v96 < 58; v96++) {	// L153
      loop29: for (int v97 = 0; v97 < 16; v97++) {	// L154
        #pragma HLS pipeline II=1
        v94[v97][v95][v96] = v93;	// L155
      }
    }
  }
  loop30: for (int v98 = 0; v98 < 58; v98++) {	// L159
    loop31: for (int v99 = 0; v99 < 58; v99++) {	// L160
      loop32: for (int v100 = 0; v100 < 16; v100++) {	// L161
        #pragma HLS pipeline II=1
        if ((v98 - 1) >= 0 && (v99 - 1) >= 0 && ((-v98) + 56) >= 0 && ((-v99) + 56) >= 0) {	// L162
          float v101 = v91[0][v100][(v98 - 1)][(v99 - 1)];	// L163
          v94[v100][v98][v99] = v101;	// L164
        }
        float v102 = v94[v100][v98][v99];	// L166
        v92.write(v102);	// L167
      }
    }
  }
  return ;	// L171
}

void forward(
  float v103[1][16][56][56],
  float v104[16][16][3][3],
  float v105[16][16][3][3],
  float v106[1][16][56][56]
) {	// L173
	#pragma HLS DATAFLOW
  hls::stream<float> v107("fifo_0");
	#pragma HLS STREAM variable=v107 depth=50176	// L177
  hls::stream<float> v108("fifo_1");
	#pragma HLS STREAM variable=v108 depth=50176	// L178
  hls::stream<float> v109("fifo_2");
	#pragma HLS STREAM variable=v109 depth=53824	// L179
  hls::stream<float> v110("fifo_3");
	#pragma HLS STREAM variable=v110 depth=50176	// L180
  hls::stream<float> v111("fifo_4");
	#pragma HLS STREAM variable=v111 depth=50176	// L181
  hls::stream<float> v112("fifo_5");
	#pragma HLS STREAM variable=v112 depth=53824	// L182
  node6(v103, v112, 0.000000);	// L183
  node5(v104, v112, v111, 0.000000);	// L184
  node4(v111, v110, 0.000010, 1.000000, 0.000000);	// L185
  node3(v110, v109, 0.000000);	// L186
  node2(v109, v105, v108, 0.000000);	// L187
  node1(v103, v108, v107, 0.000010, 0.000000, 1.000000);	// L188
  node0(v107, v106);	// L189
  return ;	// L190
}

