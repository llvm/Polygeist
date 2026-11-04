
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

void forward(
  float v0[1][8][112][112],
  float v1[8][8][1][1],
  float v2[8][3][3],
  float v3[1][8][112][112]
) {	// L9
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v0 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v3 cyclic dim=3 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=4 factor=14

  float v4[8][114][114];	// L13
  loop0: for (int v5 = 0; v5 < 8; v5++) {	// L14
    loop1: for (int v6 = 0; v6 < 57; v6++) {	// L15
      loop2: for (int v7 = 0; v7 < 114; v7++) {	// L16
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop3: for (int v8 = 0; v8 < 1; v8++) {	// L17
          loop4: for (int v9 = 0; v9 < 2; v9++) {	// L18
            loop5: for (int v10 = 0; v10 < 1; v10++) {	// L19
              v4[(v5 + v8)][(v9 + (v6 * 2))][(v7 + v10)] = 0.000000;	// L20
            }
          }
        }
      }
    }
  }
  loop6: for (int v11 = 0; v11 < 8; v11++) {	// L27
    loop7: for (int v12 = 0; v12 < 56; v12++) {	// L28
      loop8: for (int v13 = 0; v13 < 112; v13++) {	// L29
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop9: for (int v14 = 0; v14 < 1; v14++) {	// L30
          loop10: for (int v15 = 0; v15 < 2; v15++) {	// L31
            loop11: for (int v16 = 0; v16 < 1; v16++) {	// L32
              float v17 = v0[0][(v11 + v14)][(v15 + (v12 * 2))][(v13 + v16)];	// L33
              v4[(v11 + v14)][((v15 + (v12 * 2)) + 1)][((v13 + v16) + 1)] = v17;	// L34
            }
          }
        }
      }
    }
  }
  float v18[8][112][112];	// L41
  hls::stream<float> v19[1][2][14];
	#pragma HLS STREAM variable=v19 depth=3584	// L42
  float v20[8][112][112];	// L43
  loop12: for (int v21 = 0; v21 < 56; v21++) {	// L44
    loop13: for (int v22 = 0; v22 < 8; v22++) {	// L45
      loop14: for (int v23 = 0; v23 < 8; v23++) {	// L46
        loop15: for (int v24 = 0; v24 < 3; v24++) {	// L47
          loop16: for (int v25 = 0; v25 < 3; v25++) {	// L48
            #pragma HLS pipeline II=1
            #pragma HLS loop_flatten
            loop17: for (int v26 = 0; v26 < 2; v26++) {	// L49
              loop18: for (int v27 = 0; v27 < 14; v27++) {	// L50
                loop19: for (int v28 = 0; v28 < 1; v28++) {	// L51
                  loop20: for (int v29 = 0; v29 < 1; v29++) {	// L52
                    loop21: for (int v30 = 0; v30 < 1; v30++) {	// L53
                      float v31 = v4[(v23 + v28)][(((v26 + (v21 * 2)) + v24) + v29)][(((v27 + (v22 * 14)) + v25) + v30)];	// L54
                      float v32 = v2[(v23 + v28)][(v24 + v29)][(v25 + v30)];	// L55
                      if ((v24 + v29) == 0 && (v25 + v30) == 0) {	// L56
                        v18[(v23 + v28)][(v26 + (v21 * 2))][(v27 + (v22 * 14))] = 0.000000;	// L57
                      }
                      float v33 = v18[(v23 + v28)][(v26 + (v21 * 2))][(v27 + (v22 * 14))];	// L59
                      float v34 = v31 * v32;	// L60
                      float v35 = v33 + v34;	// L61
                      v18[(v23 + v28)][(v26 + (v21 * 2))][(v27 + (v22 * 14))] = v35;	// L62
                      if (((v24 + v29) - 2) == 0 && ((v25 + v30) - 2) == 0) {	// L63
                        float v36 = v18[(v23 + v28)][(v26 + (v21 * 2))][(v27 + (v22 * 14))];	// L64
                        v19[v28][v26][v27].write(v36);	// L65
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  float v37[8][112][112];	// L77
  hls::stream<float> v38[1][2][14];
	#pragma HLS STREAM variable=v38 depth=3584	// L78
  loop22: for (int v39 = 0; v39 < 8; v39++) {	// L79
    loop23: for (int v40 = 0; v40 < 56; v40++) {	// L80
      loop24: for (int v41 = 0; v41 < 8; v41++) {	// L81
        loop25: for (int v42 = 0; v42 < 8; v42++) {	// L82
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          loop26: for (int v43 = 0; v43 < 1; v43++) {	// L83
            loop27: for (int v44 = 0; v44 < 2; v44++) {	// L84
              loop28: for (int v45 = 0; v45 < 14; v45++) {	// L85
                loop29: for (int v46 = 0; v46 < 1; v46++) {	// L86
                  if ((v39 + v43) == 0) {	// L87
                    float v47 = v19[v46][v44][v45].read();	// L88
                    v20[(v42 + v46)][(v44 + (v40 * 2))][(v45 + (v41 * 14))] = v47;	// L89
                  }
                  float v48 = v20[(v42 + v46)][(v44 + (v40 * 2))][(v45 + (v41 * 14))];	// L91
                  float v49 = v1[(v39 + v43)][(v42 + v46)][0][0];	// L92
                  if ((v42 + v46) == 0) {	// L93
                    v37[(v39 + v43)][(v44 + (v40 * 2))][(v45 + (v41 * 14))] = 0.000000;	// L94
                  }
                  float v50 = v37[(v39 + v43)][(v44 + (v40 * 2))][(v45 + (v41 * 14))];	// L96
                  float v51 = v48 * v49;	// L97
                  float v52 = v50 + v51;	// L98
                  v37[(v39 + v43)][(v44 + (v40 * 2))][(v45 + (v41 * 14))] = v52;	// L99
                  if (((v42 + v46) - 7) == 0) {	// L100
                    float v53 = v37[(v39 + v43)][(v44 + (v40 * 2))][(v45 + (v41 * 14))];	// L101
                    v38[v43][v44][v45].write(v53);	// L102
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  hls::stream<float> v54[1][2][14];
	#pragma HLS STREAM variable=v54 depth=3584	// L112
  loop30: for (int v55 = 0; v55 < 8; v55++) {	// L113
    loop31: for (int v56 = 0; v56 < 56; v56++) {	// L114
      loop32: for (int v57 = 0; v57 < 8; v57++) {	// L115
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop33: for (int v58 = 0; v58 < 1; v58++) {	// L116
          loop34: for (int v59 = 0; v59 < 2; v59++) {	// L117
            loop35: for (int v60 = 0; v60 < 14; v60++) {	// L118
              float v61 = v38[v58][v59][v60].read();	// L119
              float v62 = 0.000010;	// L120
              float v63 = v62 + 1.000000;	// L121
              float v64 = 1.0 / sqrt(v63);	// L122
              float v65 = v61 * v64;	// L123
              float v66 = v65 + 0.000000;	// L124
              bool v67 = v66 > 0.000000;	// L125
              float v68 = v67 ? (float)v66 : (float)0.000000;	// L126
              v54[v58][v59][v60].write(v68);	// L127
            }
          }
        }
      }
    }
  }
  loop36: for (int v69 = 0; v69 < 8; v69++) {	// L134
    loop37: for (int v70 = 0; v70 < 56; v70++) {	// L135
      loop38: for (int v71 = 0; v71 < 8; v71++) {	// L136
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v72 = 0; v72 < 1; v72++) {	// L137
          loop40: for (int v73 = 0; v73 < 2; v73++) {	// L138
            loop41: for (int v74 = 0; v74 < 14; v74++) {	// L139
              float v75 = v54[v72][v73][v74].read();	// L140
              v3[0][(v69 + v72)][(v73 + (v70 * 2))][(v74 + (v71 * 14))] = v75;	// L141
            }
          }
        }
      }
    }
  }
  return ;	// L148
}

