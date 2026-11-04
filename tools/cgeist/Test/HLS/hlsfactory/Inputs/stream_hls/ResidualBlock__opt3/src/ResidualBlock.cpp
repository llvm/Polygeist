
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
  float v0[1][16][56][56],
  float v1[16][16][3][3],
  float v2[16][16][3][3],
  float v3[1][16][56][56]
) {	// L7
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v0 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v0 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v3 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=3 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=4 factor=7

  float v4[16][58][58];	// L11
  loop0: for (int v5 = 0; v5 < 8; v5++) {	// L12
    loop1: for (int v6 = 0; v6 < 29; v6++) {	// L13
      loop2: for (int v7 = 0; v7 < 58; v7++) {	// L14
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop3: for (int v8 = 0; v8 < 2; v8++) {	// L15
          loop4: for (int v9 = 0; v9 < 2; v9++) {	// L16
            loop5: for (int v10 = 0; v10 < 1; v10++) {	// L17
              v4[(v8 + (v5 * 2))][(v9 + (v6 * 2))][(v7 + v10)] = 0.000000;	// L18
            }
          }
        }
      }
    }
  }
  loop6: for (int v11 = 0; v11 < 8; v11++) {	// L25
    loop7: for (int v12 = 0; v12 < 28; v12++) {	// L26
      loop8: for (int v13 = 0; v13 < 56; v13++) {	// L27
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop9: for (int v14 = 0; v14 < 2; v14++) {	// L28
          loop10: for (int v15 = 0; v15 < 2; v15++) {	// L29
            loop11: for (int v16 = 0; v16 < 1; v16++) {	// L30
              float v17 = v0[0][(v14 + (v11 * 2))][(v15 + (v12 * 2))][(v13 + v16)];	// L31
              v4[(v14 + (v11 * 2))][((v15 + (v12 * 2)) + 1)][((v13 + v16) + 1)] = v17;	// L32
            }
          }
        }
      }
    }
  }
  float v18[16][56][56];	// L39
  hls::stream<float> v19[2][2][1];
	#pragma HLS STREAM variable=v19 depth=12544	// L40
  loop12: for (int v20 = 0; v20 < 8; v20++) {	// L41
    loop13: for (int v21 = 0; v21 < 28; v21++) {	// L42
      loop14: for (int v22 = 0; v22 < 56; v22++) {	// L43
        loop15: for (int v23 = 0; v23 < 8; v23++) {	// L44
          loop16: for (int v24 = 0; v24 < 3; v24++) {	// L45
            loop17: for (int v25 = 0; v25 < 3; v25++) {	// L46
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              loop18: for (int v26 = 0; v26 < 2; v26++) {	// L47
                loop19: for (int v27 = 0; v27 < 2; v27++) {	// L48
                  loop20: for (int v28 = 0; v28 < 1; v28++) {	// L49
                    loop21: for (int v29 = 0; v29 < 2; v29++) {	// L50
                      loop22: for (int v30 = 0; v30 < 1; v30++) {	// L51
                        loop23: for (int v31 = 0; v31 < 1; v31++) {	// L52
                          float v32 = v4[(v29 + (v23 * 2))][(((v27 + (v21 * 2)) + v24) + v30)][(((v22 + v28) + v25) + v31)];	// L53
                          float v33 = v1[(v26 + (v20 * 2))][(v29 + (v23 * 2))][(v24 + v30)][(v25 + v31)];	// L54
                          if ((v29 + (v23 * 2)) == 0 && (v24 + v30) == 0 && (v25 + v31) == 0) {	// L55
                            v18[(v26 + (v20 * 2))][(v27 + (v21 * 2))][(v22 + v28)] = 0.000000;	// L56
                          }
                          float v34 = v18[(v26 + (v20 * 2))][(v27 + (v21 * 2))][(v22 + v28)];	// L58
                          float v35 = v32 * v33;	// L59
                          float v36 = v34 + v35;	// L60
                          v18[(v26 + (v20 * 2))][(v27 + (v21 * 2))][(v22 + v28)] = v36;	// L61
                          if (((v29 + (v23 * 2)) - 15) == 0 && ((v24 + v30) - 2) == 0 && ((v25 + v31) - 2) == 0) {	// L62
                            float v37 = v18[(v26 + (v20 * 2))][(v27 + (v21 * 2))][(v22 + v28)];	// L63
                            v19[v26][v27][v28].write(v37);	// L64
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
    }
  }
  hls::stream<float> v38[2][2][1];
	#pragma HLS STREAM variable=v38 depth=12544	// L78
  loop24: for (int v39 = 0; v39 < 8; v39++) {	// L79
    loop25: for (int v40 = 0; v40 < 28; v40++) {	// L80
      loop26: for (int v41 = 0; v41 < 56; v41++) {	// L81
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop27: for (int v42 = 0; v42 < 2; v42++) {	// L82
          loop28: for (int v43 = 0; v43 < 2; v43++) {	// L83
            loop29: for (int v44 = 0; v44 < 1; v44++) {	// L84
              float v45 = v19[v42][v43][v44].read();	// L85
              float v46 = 0.000010;	// L86
              float v47 = v46 + 1.000000;	// L87
              float v48 = 1.0 / sqrt(v47);	// L88
              float v49 = v45 * v48;	// L89
              float v50 = v49 + 0.000000;	// L90
              bool v51 = v50 > 0.000000;	// L91
              float v52 = v51 ? (float)v50 : (float)0.000000;	// L92
              v38[v42][v43][v44].write(v52);	// L93
            }
          }
        }
      }
    }
  }
  float v53[16][58][58];	// L100
  loop30: for (int v54 = 0; v54 < 8; v54++) {	// L101
    loop31: for (int v55 = 0; v55 < 29; v55++) {	// L102
      loop32: for (int v56 = 0; v56 < 58; v56++) {	// L103
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop33: for (int v57 = 0; v57 < 2; v57++) {	// L104
          loop34: for (int v58 = 0; v58 < 2; v58++) {	// L105
            loop35: for (int v59 = 0; v59 < 1; v59++) {	// L106
              v53[(v57 + (v54 * 2))][(v58 + (v55 * 2))][(v56 + v59)] = 0.000000;	// L107
            }
          }
        }
      }
    }
  }
  loop36: for (int v60 = 0; v60 < 8; v60++) {	// L114
    loop37: for (int v61 = 0; v61 < 28; v61++) {	// L115
      loop38: for (int v62 = 0; v62 < 56; v62++) {	// L116
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v63 = 0; v63 < 2; v63++) {	// L117
          loop40: for (int v64 = 0; v64 < 2; v64++) {	// L118
            loop41: for (int v65 = 0; v65 < 1; v65++) {	// L119
              float v66 = v38[v63][v64][v65].read();	// L120
              v53[(v63 + (v60 * 2))][((v64 + (v61 * 2)) + 1)][((v62 + v65) + 1)] = v66;	// L121
            }
          }
        }
      }
    }
  }
  float v67[16][56][56];	// L128
  hls::stream<float> v68[2][2][7];
	#pragma HLS STREAM variable=v68 depth=1792	// L129
  loop42: for (int v69 = 0; v69 < 8; v69++) {	// L130
    loop43: for (int v70 = 0; v70 < 28; v70++) {	// L131
      loop44: for (int v71 = 0; v71 < 8; v71++) {	// L132
        loop45: for (int v72 = 0; v72 < 8; v72++) {	// L133
          loop46: for (int v73 = 0; v73 < 3; v73++) {	// L134
            loop47: for (int v74 = 0; v74 < 3; v74++) {	// L135
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              loop48: for (int v75 = 0; v75 < 2; v75++) {	// L136
                loop49: for (int v76 = 0; v76 < 2; v76++) {	// L137
                  loop50: for (int v77 = 0; v77 < 7; v77++) {	// L138
                    loop51: for (int v78 = 0; v78 < 2; v78++) {	// L139
                      loop52: for (int v79 = 0; v79 < 1; v79++) {	// L140
                        loop53: for (int v80 = 0; v80 < 1; v80++) {	// L141
                          float v81 = v53[(v78 + (v72 * 2))][(((v76 + (v70 * 2)) + v73) + v79)][(((v77 + (v71 * 7)) + v74) + v80)];	// L142
                          float v82 = v2[(v75 + (v69 * 2))][(v78 + (v72 * 2))][(v73 + v79)][(v74 + v80)];	// L143
                          if ((v78 + (v72 * 2)) == 0 && (v73 + v79) == 0 && (v74 + v80) == 0) {	// L144
                            v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v77 + (v71 * 7))] = 0.000000;	// L145
                          }
                          float v83 = v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v77 + (v71 * 7))];	// L147
                          float v84 = v81 * v82;	// L148
                          float v85 = v83 + v84;	// L149
                          v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v77 + (v71 * 7))] = v85;	// L150
                          if (((v78 + (v72 * 2)) - 15) == 0 && ((v73 + v79) - 2) == 0 && ((v74 + v80) - 2) == 0) {	// L151
                            float v86 = v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v77 + (v71 * 7))];	// L152
                            v68[v75][v76][v77].write(v86);	// L153
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
    }
  }
  hls::stream<float> v87[2][2][7];
	#pragma HLS STREAM variable=v87 depth=1792	// L167
  loop54: for (int v88 = 0; v88 < 8; v88++) {	// L168
    loop55: for (int v89 = 0; v89 < 28; v89++) {	// L169
      loop56: for (int v90 = 0; v90 < 8; v90++) {	// L170
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop57: for (int v91 = 0; v91 < 2; v91++) {	// L171
          loop58: for (int v92 = 0; v92 < 2; v92++) {	// L172
            loop59: for (int v93 = 0; v93 < 7; v93++) {	// L173
              float v94 = v68[v91][v92][v93].read();	// L174
              float v95 = v0[0][(v91 + (v88 * 2))][(v92 + (v89 * 2))][(v93 + (v90 * 7))];	// L175
              float v96 = 0.000010;	// L176
              float v97 = v96 + 1.000000;	// L177
              float v98 = 1.0 / sqrt(v97);	// L178
              float v99 = v94 * v98;	// L179
              float v100 = v99 + 0.000000;	// L180
              float v101 = v100 + v95;	// L181
              bool v102 = v101 > 0.000000;	// L182
              float v103 = v102 ? (float)v101 : (float)0.000000;	// L183
              v87[v91][v92][v93].write(v103);	// L184
            }
          }
        }
      }
    }
  }
  loop60: for (int v104 = 0; v104 < 8; v104++) {	// L191
    loop61: for (int v105 = 0; v105 < 28; v105++) {	// L192
      loop62: for (int v106 = 0; v106 < 8; v106++) {	// L193
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop63: for (int v107 = 0; v107 < 2; v107++) {	// L194
          loop64: for (int v108 = 0; v108 < 2; v108++) {	// L195
            loop65: for (int v109 = 0; v109 < 7; v109++) {	// L196
              float v110 = v87[v107][v108][v109].read();	// L197
              v3[0][(v107 + (v104 * 2))][(v108 + (v105 * 2))][(v109 + (v106 * 7))] = v110;	// L198
            }
          }
        }
      }
    }
  }
  return ;	// L205
}

