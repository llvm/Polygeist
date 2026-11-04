#include <iostream>
#include <iomanip>
#include <hls_stream.h>
#include "Accel.h"
#include "AccelPrint.h"

const static Word m1("0x5555555555555555", 16);
const static Word m2("0x3333333333333333", 16);
const static Word m4("0x0f0f0f0f0f0f0f0f", 16);
const static Word h01("0x0101010101010101", 16);

// -----------------------------------------------------------------------
// Hardware-specific print helpers
// -----------------------------------------------------------------------
template<typename T>
void print_ap_bits(const T& in, const unsigned W) {
  printf ("   ");
  for (unsigned i = 0; i < W; ++i)
    printf ("%3d", in[i] ? -1 : 0);
  printf ("\n");
}

template<typename T>
void print_params(T params[CONVOLVERS][K][K]) {
  for (unsigned m = 0; m < CONVOLVERS; ++m) {
    for (unsigned wr = 0; wr < K; ++wr) {
      for (unsigned wc = 0; wc < K; ++wc) {
        printf ("%3d", (params[m][wr][wc]==0) ? 0 : 1);
      }
      printf("\n");
    }
    printf("--\n");
  }
}

template<typename T>
void print_line_buffer_m(T lbuf[CONV_BANKS]) {
  for (unsigned wr = 0; wr < CONV_ROWS; ++wr) {
  for (unsigned bank = 0; bank < CONV_BANKS; ++bank) {
    for (unsigned wc = 0; wc < CONV_COLS; ++wc) {
      printf ("%3d", lbuf[bank][wr][wc].to_int());
    }
    printf (" |");
  }
  printf ("\n");
  }
}

TwoBit encode_bit(const Bit& b) {
  return (b == 0) ? TwoBit(1) : TwoBit(-1);
}

// -----------------------------------------------------------------------
// Conv
// -----------------------------------------------------------------------
ConvOut conv3x3b(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    const ap_uint<4> bank,
    const IdxType cc
) {
  ConvOut sum = 0;
  for (ap_uint<2> kr = 0; kr < K; ++kr) {
    for (ap_uint<2> kc = 0; kc < K; ++kc) {
      TwoBit data = line_buffer_m[bank][kr][cc+kc];
      const Bit& wt = conv_params_m[2-kr][2-kc];
      data[1] = (wt & data[0]) ^ data[1];
      sum += data;
    }
  }
  return sum;
}

// -----------------------------------------------------------------------
// Produce 32 elements of conv results
// -----------------------------------------------------------------------
void conv_word(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE]
) {
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    for (ap_uint<4> cc = 0; cc < BANK_WIDTH; ++cc) {
      conv_out_buffer_m[bank*BANK_WIDTH+cc] = conv3x3b( line_buffer_m, conv_params_m, bank, cc );
    }
  }
}

// -----------------------------------------------------------------------
// Process each line in a word, we need to outline this loop to
// avoid false control dependencies in Vivado HLS
// -----------------------------------------------------------------------
void process_word(
    const TwoBit  word_buffer_m[CONV_BANKS][CONV_COLS],
    const TwoBit  old_word_buffer_m[CONV_BANKS][CONV_COLS],
    const bool lb[CONV_BANKS],
    const bool rb[CONV_BANKS],
    TwoBit  line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const   Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE],
    const   ap_uint<3> log_width,
    const   ap_uint<6> words_per_image,
    const   IdxType wrd
) {
  // slices_per_line = width / BANK_WIDTH
  const ap_uint<5> slices_per_line = 1 << (log_width - LOG_BANK_WIDTH);
  const bool first_wrd = (wrd == 0);
  const bool last_wrd = (wrd == words_per_image);
  DB_PRINT(4, "process word %d, spl=%d\n", wrd.to_int(), slices_per_line.to_int());

  // Prologue
  // Update bottom row, slices are shifted left. Some slices copied from previous word (middle row)
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    ap_int<6> s_idx = bank + slices_per_line - CONV_BANKS;
    if (s_idx < 0) {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = old_word_buffer_m[CONV_BANKS+s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = lb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][CONV_COLS-1];
    } else {
      // fill from new word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = (last_wrd) ? TwoBit(0) : word_buffer_m[s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = (last_wrd || lb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = (last_wrd || rb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][CONV_COLS-1];
    }
  }
  
  DB(4,
    printf("Accel lbuf wrd%d before conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );

  // Convolution
  conv_word( line_buffer_m, conv_params_m, conv_out_buffer_m );
  
  // Update
  // Fill line buffer with lines from the new word
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    // --------------------------------------------------------------
    // Top row, slices are shifted right by slices_per_line
    ap_int<6> s_idx0 = bank - slices_per_line;
    if (s_idx0 >= 0) {
      // slice from input word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = word_buffer_m[s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][CONV_COLS-1];
    } else {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = (first_wrd) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = (first_wrd || lb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = (first_wrd || rb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][CONV_COLS-1];
    }

    // --------------------------------------------------------------
    // Middle row, simply copy the word into the line buffer
    for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
      line_buffer_m[bank][1][cc] = word_buffer_m[bank][cc];
    }
    // Fill end buffer bits
    line_buffer_m[bank][1][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[bank][0];
    line_buffer_m[bank][1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[bank][CONV_COLS-1];
  }

  DB(4,
    printf("Accel lbuf wrd%d after conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );
}

// -----------------------------------------------------------------------
// A single PE reads from all inputs and weights to generate a single
// output feature map.
// * Make sure this function gets inlined by VHLS, or cosim may fail!
// -----------------------------------------------------------------------
void bin_conv(
    Word wt_mem[CONVOLVERS][C_WT_WORDS],
    NormComp nc,
    Word dmem[2][CONVOLVERS][C_DMEM_WORDS],
    ap_uint<1> d_i_idx,
    ap_uint<1> d_o_idx,
    const unsigned   n_inputs,
    const Address    o_index,
    const ap_uint<1> new_batch,
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
) {
  const ap_uint<3> log_width = width_mode + LOG_BANK_WIDTH;
  const ap_uint<5> words_per_image = 1 << (2*width_mode);
  const unsigned n_phases = n_inputs / CONVOLVERS;
  const unsigned images_per_phase = PIX_PER_PHASE >> (2*log_width);
  const unsigned WORDS_PER_PHASE = PIX_PER_PHASE / WORD_SIZE;

  assert(n_phases % images_per_phase == 0);
  assert(n_inputs % images_per_phase == 0);
  assert(images_per_phase*words_per_image == WORDS_PER_PHASE);

  // ---------------------------------------------------------------------
  // buffers
  // ---------------------------------------------------------------------
  TwoBit  line_buffer[CONVOLVERS][CONV_BANKS][CONV_ROWS][CONV_COLS];
  Bit     conv_params[CONVOLVERS][K][K];
  ConvSum fixed_buffer[WORDS_PER_PHASE][WORD_SIZE];
  ConvSum fixed_temp[WORD_SIZE];
  // per-convolver buffers
  TwoBit  word_buffer[CONVOLVERS][CONV_BANKS][CONV_COLS];
  TwoBit  old_word_buffer[CONVOLVERS][CONV_BANKS][CONV_COLS];
  ConvOut conv_out_buffer[CONVOLVERS][WORD_SIZE];
  // edge padding flag bits
  bool lb[CONV_BANKS];
  bool rb[CONV_BANKS];

  static Address wt_addr = 0;           // address of weight word
  static ap_uint<3> wt_offset = 0;      // offset 0..6 of param
  if (new_batch != 0) { wt_addr = 0; wt_offset = 0; }

  // ---------------------------------------------------------------------
  // Calculate edge padding flag bits
  const ap_uint<4> log_slice = log_width - LOG_BANK_WIDTH;
  const ap_uint<4> w_div_8 = (1 << log_width) >> 3;
  assert (w_div_8 > 0);
  ap_uint<4> mask = ~ap_uint<4>(0);   // set mask to all 1s
  mask = mask >> (4-log_slice);
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    #pragma HLS unroll
    const ap_uint<4> x = bank & mask;
    lb[bank] = (x == 0);          // (bank % w_div_8) == 0
    rb[bank] = (x+1 == w_div_8);  // (bank % w_div_8) == w_div_8-1
  }

  // ---------------------------------------------------------------------
  // Reset conv buffer
  for (IdxType i = 0; i < WORDS_PER_PHASE; ++i) {
    for (IdxType j = 0; j < WORD_SIZE; ++j) {
      #pragma HLS UNROLL
      fixed_buffer[i][j] = 0;
    }
  }

  // ---------------------------------------------------------------------
  // Compute in phases
  // Each phase processes CONVOLVERS * WORDS_PER_PHASE input words
  // ---------------------------------------------------------------------
  LOOP_PHASES:
  for (ap_uint<10> p = 0; p < n_phases; p += images_per_phase) {
    DB(3, printf ("=== PHASE %d ===\n", p.to_int()) );

    // wrd = which word in the current image
    // wrd_phase = which wrd in the current phase
    ap_uint<8> wrd = 0;
    ap_uint<8> wrd_phase = 0;

    // Load a word each iteration, and then process it
    // We load WORDS_PER_PHASE words per phase, however we also need 1 extra "empty"
    // iteration per image in the phase to do the loop epilogue, so the loop bound
    // is WORDS_PER_PHASE + images_per_phase
    LOOP_WORDS_IN_PHASE:
    for (ap_uint<8> count = 0; count < WORDS_PER_PHASE+images_per_phase; ++count) {
      // First word of an image
      if (wrd == 0) {
        Word wt_word_buffer[CONVOLVERS];

        // -------------------------------------------------------------------
        // Load param word
        // Each word contains CONV_W_PER_WORD weight filters, after we use
        // them all we should load the next word
        // -------------------------------------------------------------------
        LOOP_WT_WORDS:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          /*if (wt_offset == 0)
            wt_word_buffer[m] = wt_mem[m][wt_addr];
          else
            wt_word_buffer[m] = wt_word_buffer[m] >> WT_SIZE;
          */
          wt_word_buffer[m] = wt_mem[m][wt_addr] >> ap_uint<6>(WT_SIZE*wt_offset);
        }
        if (wt_offset == CONV_W_PER_WORD-1) {
          ++wt_addr;
          wt_offset = 0;
        } else {
          ++wt_offset;
        }
        //print_wt_word(wt_word_buffer[0]);

        // -------------------------------------------------------------------
        // Load params
        // Each word contains CONV_W_PER_WORD weight filters packed into the first
        // 63 bits, the last bit is unused. Wts are stored in output-major order.
        // -------------------------------------------------------------------
        LOOP_LOAD_WTS:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          for (ap_uint<2> kr = 0; kr < K; ++kr) {
            for (ap_uint<2> kc = 0; kc < K; ++kc)
              conv_params[m][kr][kc] = wt_word_buffer[m][kr*K+kc];
          }
        }

        DB(3, print_params(conv_params) );
      }

      // -------------------------------------------------------------------
      // Every word in an image
      // -------------------------------------------------------------------
      // Load word
      // (wrd_phase-wrd) is which wrd in the current phase, aligned to img boundary
      if (wrd != words_per_image) {
        LOOP_CONVOLVER_LOAD:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          Word word = dmem[d_i_idx][m][p*words_per_image + wrd_phase];
          for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
            for (IdxType cc = 0; cc < CONV_COLS-2; ++cc) {
              word_buffer[m][bank][cc+1] = encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+cc)]);
            }
            word_buffer[m][bank][0          ] = (bank==0)            ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH-1)]);
            word_buffer[m][bank][CONV_COLS-1] = (bank==CONV_BANKS-1) ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+BANK_WIDTH)]);
          }
        }
      }

      // Compute
      LOOP_CONVOLVERS:
      for (IdxType m = 0; m < CONVOLVERS; ++m) {
        // Do the following for each word in an image
        process_word( word_buffer[m], old_word_buffer[m], lb, rb, line_buffer[m], conv_params[m],
            conv_out_buffer[m], log_width, words_per_image, wrd );
      } // CONVOLVERS

      for (IdxType m = 0; m < CONVOLVERS; ++m) {
        for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
          for (IdxType cc = 0; cc < CONV_COLS; ++cc) {
            old_word_buffer[m][bank][cc] = word_buffer[m][bank][cc];
        } }
      }

      // -------------------------------------------------------------------
      // Sum results across convolvers
      // -------------------------------------------------------------------
      for (IdxType i = 0; i < WORD_SIZE; ++i) {
        // Ignore conv results after processing the first word
        if (wrd > 0) {
          ConvSum s = 0;
          for (IdxType m = 0; m < CONVOLVERS; ++m)
            s += conv_out_buffer[m][i];
          fixed_buffer[wrd_phase-1][i] += s;
        }
      }

      // -------------------------------------------------------------------
      // Increment counters
      // -------------------------------------------------------------------
      if (wrd != words_per_image) {
        wrd++;
        wrd_phase++;
      } else {
        wrd = 0;
      }
    } // wrd_phase = 0 .. WORDS_PER_PHASE

  } // n_phases

  LOOP_ACC_PHASES:
  for (ap_uint<5> w = 0; w < words_per_image; ++w) {
    for (IdxType b = 0; b < WORD_SIZE; ++b) {
      #pragma HLS unroll
      fixed_temp[b] = fixed_buffer[w][b];
    }

    LOOP_ACC_PHASES_I:
    for (ap_uint<8> i = words_per_image; i < WORDS_PER_PHASE; i += words_per_image) {
      for (IdxType b = 0; b < WORD_SIZE; ++b) {
        fixed_temp[b] += fixed_buffer[w+i][b];
    } }

    for (IdxType b = 0; b < WORD_SIZE; ++b) {
      #pragma HLS unroll
      fixed_buffer[w][b] = fixed_temp[b];
    }
  }

  const Address bank_idx = o_index % CONVOLVERS;
  const Address bank_off = o_index / CONVOLVERS;
  const ap_uint<5> pool_width = 1 << (log_width-1);
  DB(4,
    unsigned width = 1 << log_width;
    printf ("=== conv result ===\n");
    print_mat(fixed_buffer[0], width, 8, width);
  );
  DB_PRINT(2, "  o_idx=%3d: nc=%6d\n", o_index.to_int(), nc.to_int());

  static Word outword;
  Word poolword;
  LOOP_BATCH_NORM:
  for (ap_uint<6> w = 0; w < words_per_image; ++w) {
    Word binword;
    Address o_bank_idx = bank_idx;
    Address o_bank_offset = bank_off*words_per_image + w;
    const ap_uint<6> out_offset = (w % 4) << 4;

    for (ap_uint<7> i = 0; i < WORD_SIZE; ++i) {
      binword[i] = (fixed_buffer[w][i] >= nc) ? 0 : 1;
    }

    if (norm_mode == 1) {
      outword = binword;
    }
    else if (norm_mode == 2) {
      // horizontal pooling first
      ap_int<WORD_SIZE/2> poolword_h;
      for (ap_uint<6> i = 0; i < WORD_SIZE/2; ++i) {
        poolword_h[i] = binword[2*i] & binword[2*i+1];
      }

      // vertical pooling
      for (ap_uint<6> i = 0; i < WORD_SIZE/4; ++i) {
        // source indices
        ap_uint<5> i0 = i >> (log_width-1);
        i0 = (i0 << log_width) + i(log_width-2,0);
        ap_uint<5> i1 = i0 + pool_width;
        // dest index
        ap_uint<6> d0 = out_offset + i;
        poolword[d0] = poolword_h[i0] & poolword_h[i1];
      }

      // For log_width > 3 we can just assign the word, but log_width = 3 means width = 8,
      // which means pooled width = 4, which is only 16 bits, which is less than 1 Word.
      // So each time we get here we only have 16 bits, meaning we have to accumulate four
      // of these 16-bit batches before writing a word out.
      if (log_width != LOG_BANK_WIDTH) {
        o_bank_offset /= 4;
        outword = poolword;
      } else {
        outword = outword >> WORD_SIZE/4;
        outword(63,48) = poolword(15,0);
        o_bank_idx = (o_index/4)%CONVOLVERS;
        o_bank_offset = (o_index/4)/CONVOLVERS;
      }
    }

    dmem[d_o_idx][o_bank_idx][o_bank_offset] = outword;
  }
}

// -----------------------------------------------------------------------
// Module to do the first conv layer
// -----------------------------------------------------------------------
void fp_conv(
    Word wt_mem[CONVOLVERS][C_WT_WORDS],
    Word kh_mem[KH_WORDS],
    Word dmem[2][CONVOLVERS][C_DMEM_WORDS],
    ap_uint<1> d_i_idx,
    ap_uint<1> d_o_idx,
    const Address kh_index,
    const Address o_index,
    const unsigned N
) {
  const unsigned M = 3;
  const unsigned S = 32;
  const unsigned OUTWORDS = 16; // words per output image

  C1InputType win[M][K][K];
  C1InputType lbuf[M][K-1][S];
  Word outwords[OUTWORDS];
  WtType wtbuf[M];

  Address wt_offset = 0;
  ap_uint<3> wt_addr = 0;

  // Parallelized across m, better for HLS
  LOOP_FP_CONV_O:
  for (IdxType n = 0; n < N; ++n) {

    // clear linebuffers for each new output map
    LOOP_RESET_LINEBUFFERS:
    for (IdxType m = 0; m < M; ++m) {
      PROLOG_COLS: for (IdxType c = 0; c < S; ++c) {
        PROLOG_ROWS: for (IdxType r = 0; r < K/2; ++r) {
          for (IdxType lr = 0; lr < K-2; ++lr) {
            lbuf[m][lr][c] = lbuf[m][lr+1][c];
          }
          lbuf[m][K-2][c] = 0;
      } }
    }

    // The weights for the 1st conv layer are just laid out
    // linearly across wt_mem, 3 weights per 64-bit word
    DB_PRINT(3, "n = %u\n", n.to_int());
    Word wt_word = wt_mem[n % CONVOLVERS][n / CONVOLVERS];
    LOOP_LOAD_WTS:
    for (ap_uint<2> m = 0; m < M; ++m) {
      wtbuf[m] = wt_word((m+1)*WT_SIZE-1, m*WT_SIZE);
      DB(3, print_wt(wtbuf[m]));
      DB(3, printf("--\n"));
    }

    // load batch norm params
    C1Comp nc;
    load_kh(nc, kh_mem, (kh_index+n));
    //printf ("  n=%3d, nc=%6.3f\n", n.to_int(), nc.to_float());

    // begin convolution
    LOOP_CONV_ROWS: for (IdxType r = 0; r < S+1; ++r) {
      LOOP_CONV_COLS: for (IdxType c = 0; c < S+1; ++c) {
        // load input word
        Word inword = 0;
        if (r < S && c < S) {
          const Address addr = r*S + c;
          inword = dmem[d_i_idx][addr/C_DMEM_WORDS][addr%C_DMEM_WORDS];
        }

        for (ap_uint<2> m = 0; m < M; ++m) {
          // load data: the value of pix is either the pixel at [r,c]
          // 0 -> +1, -1 -> -1
          // or -> 0 for padding around the boundaries
          C1InputType pix;
          const unsigned W = pix.length();
          pix(W-1,0) = inword(W-1+m*W, m*W);

          // window: shift left, leaving rightmost col for new data
          for (IdxType wr = 0; wr < K; ++wr) {
            for (IdxType wc = 0; wc < K-1; ++wc) {
              win[m][wr][wc] = win[m][wr][wc+1];
          } }

          // window: fill top K-1 pixels of rightmost column from lbuf
          for (IdxType wr = 0; wr < K-1; ++wr) {
            C1InputType val = (c != S) ? lbuf[m][wr][c] : C1InputType(0);
            win[m][wr][K-1] = val;
          }

          // window: fill bottom right with new input pixel
          win[m][K-1][K-1] = pix;

          // lbuf: shift up column c
          if (c != S) {
            for (IdxType lr = 0; lr < K-2; ++lr) {
              lbuf[m][lr][c] = lbuf[m][lr+1][c];
            }
            lbuf[m][K-2][c] = pix;
          }
        } // m

        // only perform the conv and store if legal position
        if (r > 0 && c > 0) {
          C1ConvType res = 0;
          for (ap_uint<2> m = 0; m < M; ++m) {
            for (ap_uint<2> wr = 0; wr < K; ++wr) {
              for (ap_uint<2> wc = 0; wc < K; ++wc) {
                const C1InputType& pix = win[m][wr][wc];
                const Bit& b = wtbuf[m][8-(wr*K+wc)];
                res += (b==0) ? pix : (C1InputType)(-pix);
            } }
          }

          // perform normalization right here
          outwords[(r-1)/2][((r-1)%2)*S + (c-1)] =
            (res >= nc) ? Bit(0) : Bit(-1);
        }

      } // CONV_COLS
    } // CONV_ROWS

    // Here i is the word offset within the outwords buffer
    LOOP_OUTPUT:
    for (IdxType i = 0; i < OUTWORDS; ++i) {
      Address img_idx = o_index+n;
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem[d_o_idx][bank_idx][bank_off*OUTWORDS + i] = outwords[i];
    }
  } // n
}

void bin_dense(
    const Word wt_mem[CONVOLVERS][C_WT_WORDS],
    const Word kh_mem[KH_WORDS],
    Word dmem[2][CONVOLVERS][C_DMEM_WORDS],
    ap_uint<2> layer_type,
    ap_uint<1> d_i_idx,
    ap_uint<1> d_o_idx,
    const Address o_index,
    const unsigned n_inputs,
    const unsigned n_outputs
) {
  //assert(n_outputs % WORD_SIZE == 0);
  assert(layer_type == LAYER_DENSE || n_outputs == 10);
  assert(n_inputs/WORD_SIZE % CONVOLVERS == 0);

  DenseSum sum_m[CONVOLVERS];
  // for last layer
  DenseNorm best_out = -1024;
  ap_int<8> prediction = -1;

  // read words from dmem and the wt store, dot them
  // o is the output bit, i is the input bit
  LOOP_DENSE_O:
  for (Address o = 0; o < n_outputs; ++o) {
    const Address o_addr = (o_index+o)/WORD_SIZE;
    const ap_uint<6> o_offset = (o_index+o) % WORD_SIZE;
    Word o_word = dmem[d_o_idx][o_addr%CONVOLVERS][o_addr/CONVOLVERS];

    DenseSum sum = 0;

    LOOP_DENSE_I:
    for (Address i = 0; i < n_inputs; i+=CONVOLVERS*WORD_SIZE) {
      const Address wt_addr = (o*n_inputs+i) / WORD_SIZE;

      for (IdxType j = 0; j < CONVOLVERS; ++j) {
        // in_wrd addr = [(i/WORD_SIZE+j) % CONVOLVERS][(i/WORD_SIZE+j) / CONVOLVERS]
        // wt_wrd addr = [wt_addr % CONVOLVERS][wt_addr / CONVOLVERS]
        const Word in_wrd = dmem[d_i_idx][j][i/WORD_SIZE/CONVOLVERS];
        const Word wt_wrd = wt_mem[j][wt_addr / CONVOLVERS];

        Word x = wt_wrd ^ in_wrd;

        // count_set bit for 64 bits, returns 2*cnt
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        x += x >> 8;
        x += x >> 16;
        x += x >> 32;
        x = x & 0x7f;

        sum_m[j] = WORD_SIZE - (DenseSum)(x<<1);
      }

      for (IdxType j = 0; j < CONVOLVERS; ++j)
        sum += sum_m[j];
    } // n_inputs

    // not last layer -> biniarize,
    // otherwise just store the value as a 64bit word
    if (layer_type == LAYER_DENSE) {
      Address kh_addr = o / KH_PER_WORD;
      Word kh_word = kh_mem[kh_addr];

      NormComp nc;
      IdxType kh_off = o % KH_PER_WORD;
      if (kh_off == 0)
        nc(15,0) = kh_word(15, 0);
      else if (kh_off == 1)
        nc(15,0) = kh_word(31,16);
      else if (kh_off == 2)
        nc(15,0) = kh_word(47,32);
      else
        nc(15,0) = kh_word(63,48);

      o_word[o_offset] = (sum >= nc) ? 0 : 1;
    } else {
      Address kh_addr = o / (const unsigned)2;
      Word kh_word = kh_mem[kh_addr];

      KType ki;  HType hi;
      IdxType kh_off = o % 2;
      if (kh_off == 0) {
        ki(15,0) = kh_word(15, 0);
        hi(15,0) = kh_word(31,16);
      } else {
        ki(15,0) = kh_word(47,32);
        hi(15,0) = kh_word(63,48);
      }

      //printf (" >> %d * %f + %f\n", sum.to_int(), ki.to_float(), hi.to_float());
      ap_fixed<20,10> out = ap_fixed<20,10>(sum)*ki + hi;

      if (o == 0 || out > best_out) {
        prediction = o;
        best_out = out;
      }
    }

    dmem[d_o_idx][o_addr%CONVOLVERS][o_addr/CONVOLVERS] = o_word;
  } // n_outputs

  // Here we are using o_index as a bit index, not a word index!
  if (layer_type == LAYER_LAST) {
    Word o_word;
    o_word(7,0) = prediction(7,0);
    o_word(WORD_SIZE-1, 8) = 0;
    dmem[d_o_idx][0][0] = o_word;
  }
}

// -----------------------------------------------------------------------
// Accelerator top module
// -----------------------------------------------------------------------
void top(
    Word wt_i[WT_WORDS],
    Word kh_i[KH_WORDS],
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_O_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense,last'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
) {
  DB_PRINT(2, "==== Entering Accel ====\n");
  const ap_uint<2> layer_type = layer_mode(2,1);
  const unsigned width = 8 << width_mode;
  DB_PRINT(1, "  Inputs  = %d\n", n_inputs.to_int());
  DB_PRINT(1, "  Outputs = %d\n", n_outputs.to_int());
  DB_PRINT(1, "  i_words = %d\n", input_words.to_int());
  DB_PRINT(1, "  o_words = %d\n", output_words.to_int());
  DB_PRINT(1, "  Width = %d\n", width);
  DB_PRINT(1, "  layer_mode = %d %d\n", layer_mode[0]==0 ? 0 : 1, layer_type.to_int());
  DB_PRINT(1, "  dmem_mode = %d\n", dmem_mode.to_int());

  assert(width <= MAX_WIDTH);
  assert(n_inputs != 0);
  if (layer_type <= LAYER_CONV) {
    assert(input_words % CONVOLVERS == 0);
    assert(n_inputs*width*width <= DMEM_WORDS*WORD_SIZE);
    assert(n_inputs*WT_SIZE <= WT_WORDS*WORD_SIZE);
  }

  static Word dmem[2][CONVOLVERS][C_DMEM_WORDS];
  static Word kh_mem[KH_WORDS];
  static Word wt_mem[CONVOLVERS][C_WT_WORDS];
  static Address kh_index = 0;
  static Address o_index = 0;

  if (layer_mode[0]) {
    kh_index = 0;
    o_index = 0;
  } else {
    kh_index = kh_index[0];
  }

  ap_uint<1> d_i_idx = dmem_mode;
  ap_uint<1> d_o_idx = ~dmem_mode;

  // Data input
  const ap_uint<5> words_per_image = 1 << (2*width_mode);
  Address img_idx = 0;  // i / words_per_image;
  IdxType img_off = 0;  // i % words_per_image;
  LOOP_DMEM_I: for (Address i = 0; i < input_words; ++i) {
    if (layer_type == LAYER_CONV) {
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem[d_i_idx][bank_idx][(bank_off<<(2*width_mode)) + img_off] = dmem_i[i];
    }
    else if (layer_type == LAYER_CONV1)
      dmem[d_i_idx][i/C_DMEM_WORDS][i%C_DMEM_WORDS] = dmem_i[i];
    else
      dmem[d_i_idx][i%CONVOLVERS][i/CONVOLVERS] = dmem_i[i];

    if (++img_off == words_per_image) {
      img_off = 0;
      ++img_idx;
    }
  }

  // Weight input, we must copy every 64-bit Word from the interface
  // into the accelerator
  LOOP_WT_I: for (Address i = 0; i < C_WT_WORDS*CONVOLVERS; ++i) {
    wt_mem[i%CONVOLVERS][i/CONVOLVERS] = wt_i[i];
  }
  //printf ("\nAccel Weights:\n");
  //print_params3d(wt_mem[0], 0, n_inputs*n_outputs);

  LOOP_KH_I: for (ap_uint<16> i = 0; i < KH_WORDS; ++i)
    kh_mem[i] = kh_i[i];

  if (layer_type == LAYER_CONV1) {
    assert(n_inputs == 3);

    fp_conv(
        wt_mem,
        kh_mem,
        dmem,
        d_i_idx,
        d_o_idx,
        kh_index,
        o_index,
        n_outputs
    );

    kh_index += n_outputs;
    o_index += n_outputs;
  }
  else if (layer_type == LAYER_CONV) {
    assert(norm_mode != 2 || n_outputs % 4 == 0); // needed for pooling of 8x8 image
    assert(n_inputs % CONVOLVERS == 0);

    LOOP_IMG_BATCH:
    for (IdxType i = 0; i < n_outputs; ++i) {
      // Load the batch-norm parameters for this output
      NormComp nc;
      load_kh(nc, kh_mem, kh_index);

      bin_conv(
          wt_mem,
          nc,
          dmem,
          d_i_idx, d_o_idx,
          n_inputs,
          o_index,
          i == 0 ? 1 : 0,         // new_batch
          width_mode,
          norm_mode
      );

      kh_index++;
      o_index++;
    }
  }
  else {
    bin_dense(
        wt_mem,
        kh_mem,
        dmem,
        layer_type,
        d_i_idx, d_o_idx,
        o_index,
        n_inputs, n_outputs
    );

    o_index += n_outputs;
  } // layer_type

  // Data output
  ap_uint<5> words_per_out = words_per_image / ((norm_mode!=2) ? 1 : 4);
  img_idx = 0;
  img_off = 0;
  LOOP_DMEM_O: for (Address i = 0; i < output_words; ++i) {
    // exclude conv6 (width==8, norm_mode==2) here because it writes
    // the output fmaps linearly
    if (layer_type <= LAYER_CONV && !(width_mode == 0 && norm_mode == 2)) {
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem_o[i] = dmem[d_o_idx][bank_idx][bank_off*words_per_out + img_off];
    }
    else
      dmem_o[i] = dmem[d_o_idx][i%CONVOLVERS][i/CONVOLVERS];

    if (++img_off == words_per_out) {
      img_off = 0;
      ++img_idx;
    }
  }
}
