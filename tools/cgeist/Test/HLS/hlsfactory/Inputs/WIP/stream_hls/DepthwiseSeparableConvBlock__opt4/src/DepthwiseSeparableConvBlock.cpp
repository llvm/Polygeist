
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
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v0 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v3 cyclic dim=3 factor=2
  #pragma HLS array_partition variable=v3 cyclic dim=4 factor=14

  float v4[8][114][114];
  loop0: for (int v5 = 0; v5 < 8; v5++) {
    loop1: for (int v6 = 0; v6 < 57; v6++) {
      loop2: for (int v7 = 0; v7 < 114; v7++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop3: for (int v8 = 0; v8 < 1; v8++) {
          loop4: for (int v9 = 0; v9 < 2; v9++) {
            loop5: for (int v10 = 0; v10 < 1; v10++) {
              v4[(v5 + v8)][(v9 + (v6 * 2))][(v7 + v10)] = 0.000000;
            }
          }
        }
      }
    }
  }
  loop6: for (int v11 = 0; v11 < 8; v11++) {
    loop7: for (int v12 = 0; v12 < 56; v12++) {
      loop8: for (int v13 = 0; v13 < 112; v13++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop9: for (int v14 = 0; v14 < 1; v14++) {
          loop10: for (int v15 = 0; v15 < 2; v15++) {
            loop11: for (int v16 = 0; v16 < 1; v16++) {
              float v17 = v0[0][(v11 + v14)][(v15 + (v12 * 2))][(v13 + v16)];
              v4[(v11 + v14)][((v15 + (v12 * 2)) + 1)][((v13 + v16) + 1)] = v17;
            }
          }
        }
      }
    }
  }
  float v18[8][112][112];
  hls::stream<float> v19[1][2][14];
	#pragma HLS STREAM variable=v19 depth=3584
  float v20[8][112][112];
  loop12: for (int v21 = 0; v21 < 56; v21++) {
    loop13: for (int v22 = 0; v22 < 8; v22++) {
      loop14: for (int v23 = 0; v23 < 3; v23++) {
        loop15: for (int v24 = 0; v24 < 3; v24++) {
          loop16: for (int v25 = 0; v25 < 8; v25++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_flatten
            loop17: for (int v26 = 0; v26 < 2; v26++) {
              loop18: for (int v27 = 0; v27 < 14; v27++) {
                loop19: for (int v28 = 0; v28 < 1; v28++) {
                  loop20: for (int v29 = 0; v29 < 1; v29++) {
                    loop21: for (int v30 = 0; v30 < 1; v30++) {
                      float v31 = v4[(v25 + v30)][(((v26 + (v21 * 2)) + v23) + v28)][(((v27 + (v22 * 14)) + v24) + v29)];
                      float v32 = v2[(v25 + v30)][(v23 + v28)][(v24 + v29)];
                      if ((v23 + v28) == 0 && (v24 + v29) == 0) {
                        v18[(v25 + v30)][(v26 + (v21 * 2))][(v27 + (v22 * 14))] = 0.000000;
                      }
                      float v33 = v18[(v25 + v30)][(v26 + (v21 * 2))][(v27 + (v22 * 14))];
                      float v34 = v31 * v32;
                      float v35 = v33 + v34;
                      v18[(v25 + v30)][(v26 + (v21 * 2))][(v27 + (v22 * 14))] = v35;
                      if (((v23 + v28) - 2) == 0 && ((v24 + v29) - 2) == 0) {
                        float v36 = v18[(v25 + v30)][(v26 + (v21 * 2))][(v27 + (v22 * 14))];
                        v19[v30][v26][v27].write(v36);
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
  float v37[8][112][112];
  hls::stream<float> v38[1][2][14];
	#pragma HLS STREAM variable=v38 depth=3584
  loop22: for (int v39 = 0; v39 < 56; v39++) {
    loop23: for (int v40 = 0; v40 < 8; v40++) {
      loop24: for (int v41 = 0; v41 < 8; v41++) {
        loop25: for (int v42 = 0; v42 < 8; v42++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          loop26: for (int v43 = 0; v43 < 2; v43++) {
            loop27: for (int v44 = 0; v44 < 14; v44++) {
              loop28: for (int v45 = 0; v45 < 1; v45++) {
                loop29: for (int v46 = 0; v46 < 1; v46++) {
                  if ((v42 + v46) == 0) {
                    float v47 = v19[v45][v43][v44].read();
                    v20[(v41 + v45)][(v43 + (v39 * 2))][(v44 + (v40 * 14))] = v47;
                  }
                  float v48 = v20[(v41 + v45)][(v43 + (v39 * 2))][(v44 + (v40 * 14))];
                  float v49 = v1[(v42 + v46)][(v41 + v45)][0][0];
                  if ((v41 + v45) == 0) {
                    v37[(v42 + v46)][(v43 + (v39 * 2))][(v44 + (v40 * 14))] = 0.000000;
                  }
                  float v50 = v37[(v42 + v46)][(v43 + (v39 * 2))][(v44 + (v40 * 14))];
                  float v51 = v48 * v49;
                  float v52 = v50 + v51;
                  v37[(v42 + v46)][(v43 + (v39 * 2))][(v44 + (v40 * 14))] = v52;
                  if (((v41 + v45) - 7) == 0) {
                    float v53 = v37[(v42 + v46)][(v43 + (v39 * 2))][(v44 + (v40 * 14))];
                    v38[v46][v43][v44].write(v53);
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
	#pragma HLS STREAM variable=v54 depth=3584
  loop30: for (int v55 = 0; v55 < 56; v55++) {
    loop31: for (int v56 = 0; v56 < 8; v56++) {
      loop32: for (int v57 = 0; v57 < 8; v57++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop33: for (int v58 = 0; v58 < 2; v58++) {
          loop34: for (int v59 = 0; v59 < 14; v59++) {
            loop35: for (int v60 = 0; v60 < 1; v60++) {
              float v61 = v38[v60][v58][v59].read();
              float v62 = 0.000010;
              float v63 = v62 + 1.000000;
              float v64 = 1.0 / sqrt(v63);
              float v65 = v61 * v64;
              float v66 = v65 + 0.000000;
              bool v67 = v66 > 0.000000;
              float v68 = v67 ? (float)v66 : (float)0.000000;
              v54[v60][v58][v59].write(v68);
            }
          }
        }
      }
    }
  }
  loop36: for (int v69 = 0; v69 < 56; v69++) {
    loop37: for (int v70 = 0; v70 < 8; v70++) {
      loop38: for (int v71 = 0; v71 < 8; v71++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v72 = 0; v72 < 2; v72++) {
          loop40: for (int v73 = 0; v73 < 14; v73++) {
            loop41: for (int v74 = 0; v74 < 1; v74++) {
              float v75 = v54[v74][v72][v73].read();
              v3[0][(v71 + v74)][(v72 + (v69 * 2))][(v73 + (v70 * 14))] = v75;
            }
          }
        }
      }
    }
  }
  return ;
}

