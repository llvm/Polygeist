
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
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v0 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v0 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v3 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=3 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=4 factor=7

  float v4[16][58][58];
  loop0: for (int v5 = 0; v5 < 8; v5++) {
    loop1: for (int v6 = 0; v6 < 58; v6++) {
      loop2: for (int v7 = 0; v7 < 29; v7++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop3: for (int v8 = 0; v8 < 2; v8++) {
          loop4: for (int v9 = 0; v9 < 1; v9++) {
            loop5: for (int v10 = 0; v10 < 2; v10++) {
              v4[(v8 + (v5 * 2))][(v10 + (v7 * 2))][(v6 + v9)] = 0.000000;
            }
          }
        }
      }
    }
  }
  loop6: for (int v11 = 0; v11 < 8; v11++) {
    loop7: for (int v12 = 0; v12 < 56; v12++) {
      loop8: for (int v13 = 0; v13 < 28; v13++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop9: for (int v14 = 0; v14 < 2; v14++) {
          loop10: for (int v15 = 0; v15 < 1; v15++) {
            loop11: for (int v16 = 0; v16 < 2; v16++) {
              float v17 = v0[0][(v14 + (v11 * 2))][(v16 + (v13 * 2))][(v12 + v15)];
              v4[(v14 + (v11 * 2))][((v16 + (v13 * 2)) + 1)][((v12 + v15) + 1)] = v17;
            }
          }
        }
      }
    }
  }
  float v18[16][56][56];
  hls::stream<float> v19[2][2][1];
	#pragma HLS STREAM variable=v19 depth=12544
  loop12: for (int v20 = 0; v20 < 8; v20++) {
    loop13: for (int v21 = 0; v21 < 8; v21++) {
      loop14: for (int v22 = 0; v22 < 28; v22++) {
        loop15: for (int v23 = 0; v23 < 3; v23++) {
          loop16: for (int v24 = 0; v24 < 3; v24++) {
            loop17: for (int v25 = 0; v25 < 56; v25++) {
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              loop18: for (int v26 = 0; v26 < 2; v26++) {
                loop19: for (int v27 = 0; v27 < 2; v27++) {
                  loop20: for (int v28 = 0; v28 < 2; v28++) {
                    loop21: for (int v29 = 0; v29 < 1; v29++) {
                      loop22: for (int v30 = 0; v30 < 1; v30++) {
                        loop23: for (int v31 = 0; v31 < 1; v31++) {
                          float v32 = v4[(v27 + (v21 * 2))][(((v28 + (v22 * 2)) + v23) + v29)][(((v25 + v31) + v24) + v30)];
                          float v33 = v1[(v26 + (v20 * 2))][(v27 + (v21 * 2))][(v23 + v29)][(v24 + v30)];
                          if ((v27 + (v21 * 2)) == 0 && (v23 + v29) == 0 && (v24 + v30) == 0) {
                            v18[(v26 + (v20 * 2))][(v28 + (v22 * 2))][(v25 + v31)] = 0.000000;
                          }
                          float v34 = v18[(v26 + (v20 * 2))][(v28 + (v22 * 2))][(v25 + v31)];
                          float v35 = v32 * v33;
                          float v36 = v34 + v35;
                          v18[(v26 + (v20 * 2))][(v28 + (v22 * 2))][(v25 + v31)] = v36;
                          if (((v27 + (v21 * 2)) - 15) == 0 && ((v23 + v29) - 2) == 0 && ((v24 + v30) - 2) == 0) {
                            float v37 = v18[(v26 + (v20 * 2))][(v28 + (v22 * 2))][(v25 + v31)];
                            v19[v26][v28][v31].write(v37);
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
	#pragma HLS STREAM variable=v38 depth=12544
  loop24: for (int v39 = 0; v39 < 8; v39++) {
    loop25: for (int v40 = 0; v40 < 28; v40++) {
      loop26: for (int v41 = 0; v41 < 56; v41++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop27: for (int v42 = 0; v42 < 2; v42++) {
          loop28: for (int v43 = 0; v43 < 2; v43++) {
            loop29: for (int v44 = 0; v44 < 1; v44++) {
              float v45 = v19[v42][v43][v44].read();
              float v46 = 0.000010;
              float v47 = v46 + 1.000000;
              float v48 = 1.0 / sqrt(v47);
              float v49 = v45 * v48;
              float v50 = v49 + 0.000000;
              bool v51 = v50 > 0.000000;
              float v52 = v51 ? (float)v50 : (float)0.000000;
              v38[v42][v43][v44].write(v52);
            }
          }
        }
      }
    }
  }
  float v53[16][58][58];
  loop30: for (int v54 = 0; v54 < 8; v54++) {
    loop31: for (int v55 = 0; v55 < 58; v55++) {
      loop32: for (int v56 = 0; v56 < 29; v56++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop33: for (int v57 = 0; v57 < 2; v57++) {
          loop34: for (int v58 = 0; v58 < 1; v58++) {
            loop35: for (int v59 = 0; v59 < 2; v59++) {
              v53[(v57 + (v54 * 2))][(v59 + (v56 * 2))][(v55 + v58)] = 0.000000;
            }
          }
        }
      }
    }
  }
  loop36: for (int v60 = 0; v60 < 8; v60++) {
    loop37: for (int v61 = 0; v61 < 28; v61++) {
      loop38: for (int v62 = 0; v62 < 56; v62++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v63 = 0; v63 < 2; v63++) {
          loop40: for (int v64 = 0; v64 < 2; v64++) {
            loop41: for (int v65 = 0; v65 < 1; v65++) {
              float v66 = v38[v63][v64][v65].read();
              v53[(v63 + (v60 * 2))][((v64 + (v61 * 2)) + 1)][((v62 + v65) + 1)] = v66;
            }
          }
        }
      }
    }
  }
  float v67[16][56][56];
  hls::stream<float> v68[2][2][7];
	#pragma HLS STREAM variable=v68 depth=1792
  loop42: for (int v69 = 0; v69 < 8; v69++) {
    loop43: for (int v70 = 0; v70 < 28; v70++) {
      loop44: for (int v71 = 0; v71 < 8; v71++) {
        loop45: for (int v72 = 0; v72 < 3; v72++) {
          loop46: for (int v73 = 0; v73 < 3; v73++) {
            loop47: for (int v74 = 0; v74 < 8; v74++) {
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              loop48: for (int v75 = 0; v75 < 2; v75++) {
                loop49: for (int v76 = 0; v76 < 2; v76++) {
                  loop50: for (int v77 = 0; v77 < 2; v77++) {
                    loop51: for (int v78 = 0; v78 < 1; v78++) {
                      loop52: for (int v79 = 0; v79 < 1; v79++) {
                        loop53: for (int v80 = 0; v80 < 7; v80++) {
                          float v81 = v53[(v77 + (v71 * 2))][(((v76 + (v70 * 2)) + v72) + v78)][(((v80 + (v74 * 7)) + v73) + v79)];
                          float v82 = v2[(v75 + (v69 * 2))][(v77 + (v71 * 2))][(v72 + v78)][(v73 + v79)];
                          if ((v77 + (v71 * 2)) == 0 && (v72 + v78) == 0 && (v73 + v79) == 0) {
                            v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v80 + (v74 * 7))] = 0.000000;
                          }
                          float v83 = v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v80 + (v74 * 7))];
                          float v84 = v81 * v82;
                          float v85 = v83 + v84;
                          v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v80 + (v74 * 7))] = v85;
                          if (((v77 + (v71 * 2)) - 15) == 0 && ((v72 + v78) - 2) == 0 && ((v73 + v79) - 2) == 0) {
                            float v86 = v67[(v75 + (v69 * 2))][(v76 + (v70 * 2))][(v80 + (v74 * 7))];
                            v68[v75][v76][v80].write(v86);
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
	#pragma HLS STREAM variable=v87 depth=1792
  loop54: for (int v88 = 0; v88 < 8; v88++) {
    loop55: for (int v89 = 0; v89 < 28; v89++) {
      loop56: for (int v90 = 0; v90 < 8; v90++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop57: for (int v91 = 0; v91 < 2; v91++) {
          loop58: for (int v92 = 0; v92 < 2; v92++) {
            loop59: for (int v93 = 0; v93 < 7; v93++) {
              float v94 = v68[v91][v92][v93].read();
              float v95 = v0[0][(v91 + (v88 * 2))][(v92 + (v89 * 2))][(v93 + (v90 * 7))];
              float v96 = 0.000010;
              float v97 = v96 + 1.000000;
              float v98 = 1.0 / sqrt(v97);
              float v99 = v94 * v98;
              float v100 = v99 + 0.000000;
              float v101 = v100 + v95;
              bool v102 = v101 > 0.000000;
              float v103 = v102 ? (float)v101 : (float)0.000000;
              v87[v91][v92][v93].write(v103);
            }
          }
        }
      }
    }
  }
  loop60: for (int v104 = 0; v104 < 8; v104++) {
    loop61: for (int v105 = 0; v105 < 28; v105++) {
      loop62: for (int v106 = 0; v106 < 8; v106++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop63: for (int v107 = 0; v107 < 2; v107++) {
          loop64: for (int v108 = 0; v108 < 2; v108++) {
            loop65: for (int v109 = 0; v109 < 7; v109++) {
              float v110 = v87[v107][v108][v109].read();
              v3[0][(v107 + (v104 * 2))][(v108 + (v105 * 2))][(v109 + (v106 * 7))] = v110;
            }
          }
        }
      }
    }
  }
  return ;
}

