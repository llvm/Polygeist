#ifndef ACCEL_ACCEL_SCHEDULE_H
#define ACCEL_ACCEL_SCHEDULE_H

#include <vector>
#include "Accel.h"

// Contains all info needed to invoke the accelerator once except
// input/output data and its size which is handled separately
struct AccelInfo {
  Word* wt;
  Word* kh;
  unsigned n_inputs;
  unsigned n_outputs;
  ap_uint<3> layer_mode;  // [0]='new layer', [2:1]='conv1,conv,dense'
  ap_uint<2> width_mode;  // 0=8'b, 1=16'b, 2=32'b
  ap_uint<2> norm_mode;   // 0='do nothing', 1='do norm', 2='do pool'

  AccelInfo() {
    wt = new Word[WT_WORDS];
    kh = new Word[KH_WORDS];
  }

  ~AccelInfo() {
    delete[] wt;
    delete[] kh;
  }
};

typedef std::vector<AccelInfo> AccelSchedule;

void compute_accel_schedule(
    Word* wt,
    Word* kh,
    unsigned n_inputs,
    unsigned n_outputs,
    unsigned width,
    const ap_uint<2> layer_type,  // 0=conv1, 1=conv, 2=dense
    const ap_uint<1> max_pool,
    AccelSchedule &schedule
);

void run_accel_schedule(
    Word* data_i,
    Word* data_o,
    unsigned layer_idx,
    unsigned input_words,
    unsigned output_words,
    ap_uint<1> dmem_mode,
    AccelSchedule& s
);

void load_conv1_weights(Word* wt, Word* wt_o,
                  unsigned o, unsigned n_out);
void load_conv_weights(Word* wt, Word* wt_o,
                  unsigned o, unsigned n_in, unsigned n_out);
void load_dense_weights(Word* wt, Word* wt_o,
                  unsigned o, unsigned n_in, unsigned n_out);

void load_kh(Word* kh, Word* kh_mem, unsigned o, unsigned n_imgs);

unsigned find_conv_batch_size(unsigned width, unsigned width_o,
                         unsigned n_inputs, unsigned n_outputs);
unsigned find_dense_batch_size(unsigned n_inputs, unsigned n_outputs);

float total_time();

#endif
