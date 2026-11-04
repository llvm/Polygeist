// this file can be generated from the following SODA DSL
/*
kernel: erosion
burst width: 64
iterate: 1
unroll factor: 4
input dram 0 int16: input(480, *)
local int16: tmp(0, 9) = min(input(0, 0), input(0, 1), input(0, 2), input(0, 3), input(0, 4), input(0, 5), input(0, 6), input(0, 7), input(0, 8), input(0, 9), input(0, 10), input(0, 11), input(0, 12), input(0, 13), input(0, 14), input(0, 15), input(0, 16), input(0, 17), input(0, 18))
output dram 1 int16: output(9, 0) = min(tmp(0, 0), tmp(1, 0), tmp(2, 0), tmp(3, 0), tmp(4, 0), tmp(5, 0), tmp(6, 0), tmp(7, 0), tmp(8, 0), tmp(9, 0), tmp(10, 0), tmp(11, 0), tmp(12, 0), tmp(13, 0), tmp(14, 0), tmp(15, 0), tmp(16, 0), tmp(17, 0), tmp(18, 0))
border: ignore
cluster: none
*/

// stencil window size: (19, 19)
// stencil distace: 8658
// data layout is documented at
// https://github.com/Blaok/soda/blob/master/docs/data-layout.md

#include <cfloat>
#include <cmath>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <tuple>

#include <ap_int.h>
#include <hls_stream.h>

#ifdef __SYNTHESIS__
#warning this file should be used for simulation only
#warning synthesis result may be sub-optimal
#endif  // __SYNTHESIS__

template<typename To, typename From>
To Reinterpret(From val) {
#pragma HLS inline
  return reinterpret_cast<To&>(val);
}

template<typename T>
struct Data {
  T data;
  bool ctrl;
};

template<typename T>
bool ReadData(T& data, hls::stream<Data<T>>& from) {
#pragma HLS inline
  const auto tmp = from.read();
  data = tmp.data;
  return tmp.ctrl;
}

template<typename T>
void WriteData(hls::stream<Data<T>>& to, const T& data, bool ctrl) {
#pragma HLS inline
  Data<T> tmp;
  tmp.data = data;
  tmp.ctrl = ctrl;
  to.write(tmp);
}

template <typename T>
void BurstRead(hls::stream<Data<T>>& to, T* from, uint64_t data_num) {
load:
  for (uint64_t i = 0; i < data_num;) {
#pragma HLS pipeline II = 1
    const uint64_t next_i = i + 1;
    WriteData(to, from[i], next_i < data_num);
    i = next_i;
  }
}

template <typename T>
void BurstWrite(T* to, hls::stream<Data<T>>& from, uint64_t data_num) {
store:
  for (uint64_t i = 0; i < data_num; ++i) {
#pragma HLS pipeline II = 1
    T buf;
    ReadData(buf, from);
    to[i] = buf;
  }
}

void Module0Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /* input*/ hls::stream<Data<ap_uint<64>>>& dram_input_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_input_bank_0_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_0:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!dram_input_bank_0_fifo.empty())
    {
      ap_uint<64> dram_input_bank_0_buf;
      const bool dram_input_bank_0_buf_enable = ReadData(dram_input_bank_0_buf, dram_input_bank_0_fifo);
      const bool enabled = dram_input_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(63, 48))), enabled);
      WriteData(fifo_st_1, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(47, 32))), enabled);
      WriteData(fifo_st_2, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(31, 16))), enabled);
      WriteData(fifo_st_3, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(15, 0))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_1:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<7> ptr_delay_120 = 0;
  int16_t fifo_ref_0_delayed_120_buf[120];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_120_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_2:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_120_buf distance = 120
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_120 = fifo_ref_0_delayed_120_buf[ptr_delay_120];;
      const int16_t let_0 = fifo_ref_0_delayed_120;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      fifo_ref_0_delayed_120_buf[ptr_delay_120] = fifo_ref_0;
      ptr_delay_120 < 119 ? (++ptr_delay_120) : (ptr_delay_120 = 0);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<7> ptr_delay_120 = 0;
  int16_t fifo_ref_0_delayed_120_buf[120];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_120_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_3:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_120_buf distance = 120
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_120 = fifo_ref_0_delayed_120_buf[ptr_delay_120];;
      const int16_t let_0 = fifo_ref_0_delayed_120;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      fifo_ref_0_delayed_120_buf[ptr_delay_120] = fifo_ref_0;
      ptr_delay_120 < 119 ? (++ptr_delay_120) : (ptr_delay_120 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_6, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_7, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_8, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_9, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_10, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_11, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_12, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_13, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_14, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_15, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_16, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_17, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_18)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#pragma HLS aggregate bit variable = fifo_ld_4
#pragma HLS aggregate bit variable = fifo_ld_5
#pragma HLS aggregate bit variable = fifo_ld_6
#pragma HLS aggregate bit variable = fifo_ld_7
#pragma HLS aggregate bit variable = fifo_ld_8
#pragma HLS aggregate bit variable = fifo_ld_9
#pragma HLS aggregate bit variable = fifo_ld_10
#pragma HLS aggregate bit variable = fifo_ld_11
#pragma HLS aggregate bit variable = fifo_ld_12
#pragma HLS aggregate bit variable = fifo_ld_13
#pragma HLS aggregate bit variable = fifo_ld_14
#pragma HLS aggregate bit variable = fifo_ld_15
#pragma HLS aggregate bit variable = fifo_ld_16
#pragma HLS aggregate bit variable = fifo_ld_17
#pragma HLS aggregate bit variable = fifo_ld_18
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty() && !fifo_ld_7.empty() && !fifo_ld_8.empty() && !fifo_ld_9.empty() && !fifo_ld_10.empty() && !fifo_ld_11.empty() && !fifo_ld_12.empty() && !fifo_ld_13.empty() && !fifo_ld_14.empty() && !fifo_ld_15.empty() && !fifo_ld_16.empty() && !fifo_ld_17.empty() && !fifo_ld_18.empty())
    {
      int16_t fifo_ref_0;
      int16_t fifo_ref_1;
      int16_t fifo_ref_2;
      int16_t fifo_ref_3;
      int16_t fifo_ref_4;
      int16_t fifo_ref_5;
      int16_t fifo_ref_6;
      int16_t fifo_ref_7;
      int16_t fifo_ref_8;
      int16_t fifo_ref_9;
      int16_t fifo_ref_10;
      int16_t fifo_ref_11;
      int16_t fifo_ref_12;
      int16_t fifo_ref_13;
      int16_t fifo_ref_14;
      int16_t fifo_ref_15;
      int16_t fifo_ref_16;
      int16_t fifo_ref_17;
      int16_t fifo_ref_18;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool fifo_ref_6_enable = ReadData(fifo_ref_6, fifo_ld_6);
      const bool fifo_ref_7_enable = ReadData(fifo_ref_7, fifo_ld_7);
      const bool fifo_ref_8_enable = ReadData(fifo_ref_8, fifo_ld_8);
      const bool fifo_ref_9_enable = ReadData(fifo_ref_9, fifo_ld_9);
      const bool fifo_ref_10_enable = ReadData(fifo_ref_10, fifo_ld_10);
      const bool fifo_ref_11_enable = ReadData(fifo_ref_11, fifo_ld_11);
      const bool fifo_ref_12_enable = ReadData(fifo_ref_12, fifo_ld_12);
      const bool fifo_ref_13_enable = ReadData(fifo_ref_13, fifo_ld_13);
      const bool fifo_ref_14_enable = ReadData(fifo_ref_14, fifo_ld_14);
      const bool fifo_ref_15_enable = ReadData(fifo_ref_15, fifo_ld_15);
      const bool fifo_ref_16_enable = ReadData(fifo_ref_16, fifo_ld_16);
      const bool fifo_ref_17_enable = ReadData(fifo_ref_17, fifo_ld_17);
      const bool fifo_ref_18_enable = ReadData(fifo_ref_18, fifo_ld_18);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable && fifo_ref_7_enable && fifo_ref_8_enable && fifo_ref_9_enable && fifo_ref_10_enable && fifo_ref_11_enable && fifo_ref_12_enable && fifo_ref_13_enable && fifo_ref_14_enable && fifo_ref_15_enable && fifo_ref_16_enable && fifo_ref_17_enable && fifo_ref_18_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t((int16_t)(std::min(std::min(std::min(std::min(fifo_ref_0, fifo_ref_1), std::min(fifo_ref_2, fifo_ref_3)), std::min(std::min(fifo_ref_4, fifo_ref_5), std::min(fifo_ref_6, std::min(fifo_ref_7, fifo_ref_8)))), std::min(std::min(std::min(fifo_ref_9, fifo_ref_10), std::min(fifo_ref_11, std::min(fifo_ref_12, fifo_ref_13))), std::min(std::min(fifo_ref_14, fifo_ref_15), std::min(fifo_ref_16, std::min(fifo_ref_17, fifo_ref_18))))))), enabled);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_4, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  int16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_5:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const int16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      WriteData(fifo_st_3, int16_t(let_0), enabled);
      WriteData(fifo_st_4, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  int16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_6:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const int16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_7:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_2, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  int16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const int16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_9:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_2, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_3, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  int16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_10:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const int16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      WriteData(fifo_st_3, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_10
} // Module10Func

void Module11Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_4, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_11:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_2, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_3, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_4, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_11
} // Module11Func

void Module12Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  int16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_12:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const int16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_12
} // Module12Func

void Module13Func(
  /*output*/ hls::stream<Data<ap_uint<64>>>& dram_output_bank_1_fifo, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = dram_output_bank_1_fifo
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_13:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      int16_t fifo_ref_0;
      int16_t fifo_ref_1;
      int16_t fifo_ref_2;
      int16_t fifo_ref_3;
      ap_uint<64> dram_output_bank_1_buf;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      dram_output_bank_1_buf(63, 48) = Reinterpret<ap_uint<16>>(fifo_ref_0);
      dram_output_bank_1_buf(47, 32) = Reinterpret<ap_uint<16>>(fifo_ref_1);
      dram_output_bank_1_buf(31, 16) = Reinterpret<ap_uint<16>>(fifo_ref_2);
      dram_output_bank_1_buf(15, 0) = Reinterpret<ap_uint<16>>(fifo_ref_3);
      WriteData(dram_output_bank_1_fifo, dram_output_bank_1_buf, enabled);
    } // if not empty
  } // for module_13
} // Module13Func

extern "C" {

void erosion_kernel(
  ap_uint<64>* bank_1_output, 
  ap_uint<64>* bank_0_input, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_output offset = slave bundle = output_bank_1
#pragma HLS interface m_axi port = bank_0_input offset = slave bundle = input_bank_0
#pragma HLS interface s_axilite port = bank_1_output bundle = control
#pragma HLS interface s_axilite port = bank_0_input bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<64>>> bank_0_input_buf("bank_0_input_buf");
#pragma HLS stream variable = bank_0_input_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_input_buf
  hls::stream<Data<ap_uint<64>>> bank_1_output_buf("bank_1_output_buf");
#pragma HLS stream variable = bank_1_output_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_output_buf

  hls::stream<Data<int16_t>> from_input_bank_0_to_input_offset_0("from_input_bank_0_to_input_offset_0");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_0
  hls::stream<Data<int16_t>> from_input_bank_0_to_input_offset_1("from_input_bank_0_to_input_offset_1");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_1
  hls::stream<Data<int16_t>> from_input_bank_0_to_input_offset_2("from_input_bank_0_to_input_offset_2");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_2
  hls::stream<Data<int16_t>> from_input_bank_0_to_input_offset_3("from_input_bank_0_to_input_offset_3");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_3
  hls::stream<Data<int16_t>> from_input_offset_0_to_input_offset_480("from_input_offset_0_to_input_offset_480");
#pragma HLS stream variable = from_input_offset_0_to_input_offset_480 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_0_to_input_offset_480
  hls::stream<Data<int16_t>> from_input_offset_0_to_tmp_pe_3("from_input_offset_0_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_0_to_tmp_pe_3 depth = 39
#pragma HLS aggregate bit variable = from_input_offset_0_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_1_to_input_offset_481("from_input_offset_1_to_input_offset_481");
#pragma HLS stream variable = from_input_offset_1_to_input_offset_481 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1_to_input_offset_481
  hls::stream<Data<int16_t>> from_input_offset_1_to_tmp_pe_2("from_input_offset_1_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_1_to_tmp_pe_2 depth = 39
#pragma HLS aggregate bit variable = from_input_offset_1_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_2_to_input_offset_482("from_input_offset_2_to_input_offset_482");
#pragma HLS stream variable = from_input_offset_2_to_input_offset_482 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2_to_input_offset_482
  hls::stream<Data<int16_t>> from_input_offset_2_to_tmp_pe_1("from_input_offset_2_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_2_to_tmp_pe_1 depth = 39
#pragma HLS aggregate bit variable = from_input_offset_2_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_3_to_input_offset_483("from_input_offset_3_to_input_offset_483");
#pragma HLS stream variable = from_input_offset_3_to_input_offset_483 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3_to_input_offset_483
  hls::stream<Data<int16_t>> from_input_offset_3_to_tmp_pe_0("from_input_offset_3_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_3_to_tmp_pe_0 depth = 39
#pragma HLS aggregate bit variable = from_input_offset_3_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_480_to_input_offset_960("from_input_offset_480_to_input_offset_960");
#pragma HLS stream variable = from_input_offset_480_to_input_offset_960 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_480_to_input_offset_960
  hls::stream<Data<int16_t>> from_input_offset_480_to_tmp_pe_3("from_input_offset_480_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_480_to_tmp_pe_3 depth = 37
#pragma HLS aggregate bit variable = from_input_offset_480_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_481_to_input_offset_961("from_input_offset_481_to_input_offset_961");
#pragma HLS stream variable = from_input_offset_481_to_input_offset_961 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_481_to_input_offset_961
  hls::stream<Data<int16_t>> from_input_offset_481_to_tmp_pe_2("from_input_offset_481_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_481_to_tmp_pe_2 depth = 37
#pragma HLS aggregate bit variable = from_input_offset_481_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_482_to_input_offset_962("from_input_offset_482_to_input_offset_962");
#pragma HLS stream variable = from_input_offset_482_to_input_offset_962 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_482_to_input_offset_962
  hls::stream<Data<int16_t>> from_input_offset_482_to_tmp_pe_1("from_input_offset_482_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_482_to_tmp_pe_1 depth = 37
#pragma HLS aggregate bit variable = from_input_offset_482_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_483_to_input_offset_963("from_input_offset_483_to_input_offset_963");
#pragma HLS stream variable = from_input_offset_483_to_input_offset_963 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_483_to_input_offset_963
  hls::stream<Data<int16_t>> from_input_offset_483_to_tmp_pe_0("from_input_offset_483_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_483_to_tmp_pe_0 depth = 37
#pragma HLS aggregate bit variable = from_input_offset_483_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_960_to_input_offset_1440("from_input_offset_960_to_input_offset_1440");
#pragma HLS stream variable = from_input_offset_960_to_input_offset_1440 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_960_to_input_offset_1440
  hls::stream<Data<int16_t>> from_input_offset_960_to_tmp_pe_3("from_input_offset_960_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_960_to_tmp_pe_3 depth = 35
#pragma HLS aggregate bit variable = from_input_offset_960_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_961_to_input_offset_1441("from_input_offset_961_to_input_offset_1441");
#pragma HLS stream variable = from_input_offset_961_to_input_offset_1441 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_961_to_input_offset_1441
  hls::stream<Data<int16_t>> from_input_offset_961_to_tmp_pe_2("from_input_offset_961_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_961_to_tmp_pe_2 depth = 35
#pragma HLS aggregate bit variable = from_input_offset_961_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_962_to_input_offset_1442("from_input_offset_962_to_input_offset_1442");
#pragma HLS stream variable = from_input_offset_962_to_input_offset_1442 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_962_to_input_offset_1442
  hls::stream<Data<int16_t>> from_input_offset_962_to_tmp_pe_1("from_input_offset_962_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_962_to_tmp_pe_1 depth = 35
#pragma HLS aggregate bit variable = from_input_offset_962_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_963_to_input_offset_1443("from_input_offset_963_to_input_offset_1443");
#pragma HLS stream variable = from_input_offset_963_to_input_offset_1443 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_963_to_input_offset_1443
  hls::stream<Data<int16_t>> from_input_offset_963_to_tmp_pe_0("from_input_offset_963_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_963_to_tmp_pe_0 depth = 35
#pragma HLS aggregate bit variable = from_input_offset_963_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_1440_to_input_offset_1920("from_input_offset_1440_to_input_offset_1920");
#pragma HLS stream variable = from_input_offset_1440_to_input_offset_1920 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1440_to_input_offset_1920
  hls::stream<Data<int16_t>> from_input_offset_1440_to_tmp_pe_3("from_input_offset_1440_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_1440_to_tmp_pe_3 depth = 33
#pragma HLS aggregate bit variable = from_input_offset_1440_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_1441_to_input_offset_1921("from_input_offset_1441_to_input_offset_1921");
#pragma HLS stream variable = from_input_offset_1441_to_input_offset_1921 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1441_to_input_offset_1921
  hls::stream<Data<int16_t>> from_input_offset_1441_to_tmp_pe_2("from_input_offset_1441_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_1441_to_tmp_pe_2 depth = 33
#pragma HLS aggregate bit variable = from_input_offset_1441_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_1442_to_input_offset_1922("from_input_offset_1442_to_input_offset_1922");
#pragma HLS stream variable = from_input_offset_1442_to_input_offset_1922 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1442_to_input_offset_1922
  hls::stream<Data<int16_t>> from_input_offset_1442_to_tmp_pe_1("from_input_offset_1442_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_1442_to_tmp_pe_1 depth = 33
#pragma HLS aggregate bit variable = from_input_offset_1442_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_1443_to_input_offset_1923("from_input_offset_1443_to_input_offset_1923");
#pragma HLS stream variable = from_input_offset_1443_to_input_offset_1923 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1443_to_input_offset_1923
  hls::stream<Data<int16_t>> from_input_offset_1443_to_tmp_pe_0("from_input_offset_1443_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_1443_to_tmp_pe_0 depth = 33
#pragma HLS aggregate bit variable = from_input_offset_1443_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_1920_to_input_offset_2400("from_input_offset_1920_to_input_offset_2400");
#pragma HLS stream variable = from_input_offset_1920_to_input_offset_2400 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1920_to_input_offset_2400
  hls::stream<Data<int16_t>> from_input_offset_1920_to_tmp_pe_3("from_input_offset_1920_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_1920_to_tmp_pe_3 depth = 31
#pragma HLS aggregate bit variable = from_input_offset_1920_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_1921_to_input_offset_2401("from_input_offset_1921_to_input_offset_2401");
#pragma HLS stream variable = from_input_offset_1921_to_input_offset_2401 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1921_to_input_offset_2401
  hls::stream<Data<int16_t>> from_input_offset_1921_to_tmp_pe_2("from_input_offset_1921_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_1921_to_tmp_pe_2 depth = 31
#pragma HLS aggregate bit variable = from_input_offset_1921_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_1922_to_input_offset_2402("from_input_offset_1922_to_input_offset_2402");
#pragma HLS stream variable = from_input_offset_1922_to_input_offset_2402 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1922_to_input_offset_2402
  hls::stream<Data<int16_t>> from_input_offset_1922_to_tmp_pe_1("from_input_offset_1922_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_1922_to_tmp_pe_1 depth = 31
#pragma HLS aggregate bit variable = from_input_offset_1922_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_1923_to_input_offset_2403("from_input_offset_1923_to_input_offset_2403");
#pragma HLS stream variable = from_input_offset_1923_to_input_offset_2403 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1923_to_input_offset_2403
  hls::stream<Data<int16_t>> from_input_offset_1923_to_tmp_pe_0("from_input_offset_1923_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_1923_to_tmp_pe_0 depth = 31
#pragma HLS aggregate bit variable = from_input_offset_1923_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_2400_to_input_offset_2880("from_input_offset_2400_to_input_offset_2880");
#pragma HLS stream variable = from_input_offset_2400_to_input_offset_2880 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2400_to_input_offset_2880
  hls::stream<Data<int16_t>> from_input_offset_2400_to_tmp_pe_3("from_input_offset_2400_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_2400_to_tmp_pe_3 depth = 29
#pragma HLS aggregate bit variable = from_input_offset_2400_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_2401_to_input_offset_2881("from_input_offset_2401_to_input_offset_2881");
#pragma HLS stream variable = from_input_offset_2401_to_input_offset_2881 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2401_to_input_offset_2881
  hls::stream<Data<int16_t>> from_input_offset_2401_to_tmp_pe_2("from_input_offset_2401_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_2401_to_tmp_pe_2 depth = 29
#pragma HLS aggregate bit variable = from_input_offset_2401_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_2402_to_input_offset_2882("from_input_offset_2402_to_input_offset_2882");
#pragma HLS stream variable = from_input_offset_2402_to_input_offset_2882 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2402_to_input_offset_2882
  hls::stream<Data<int16_t>> from_input_offset_2402_to_tmp_pe_1("from_input_offset_2402_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_2402_to_tmp_pe_1 depth = 29
#pragma HLS aggregate bit variable = from_input_offset_2402_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_2403_to_input_offset_2883("from_input_offset_2403_to_input_offset_2883");
#pragma HLS stream variable = from_input_offset_2403_to_input_offset_2883 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2403_to_input_offset_2883
  hls::stream<Data<int16_t>> from_input_offset_2403_to_tmp_pe_0("from_input_offset_2403_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_2403_to_tmp_pe_0 depth = 29
#pragma HLS aggregate bit variable = from_input_offset_2403_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_2880_to_input_offset_3360("from_input_offset_2880_to_input_offset_3360");
#pragma HLS stream variable = from_input_offset_2880_to_input_offset_3360 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2880_to_input_offset_3360
  hls::stream<Data<int16_t>> from_input_offset_2880_to_tmp_pe_3("from_input_offset_2880_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_2880_to_tmp_pe_3 depth = 27
#pragma HLS aggregate bit variable = from_input_offset_2880_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_2881_to_input_offset_3361("from_input_offset_2881_to_input_offset_3361");
#pragma HLS stream variable = from_input_offset_2881_to_input_offset_3361 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2881_to_input_offset_3361
  hls::stream<Data<int16_t>> from_input_offset_2881_to_tmp_pe_2("from_input_offset_2881_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_2881_to_tmp_pe_2 depth = 27
#pragma HLS aggregate bit variable = from_input_offset_2881_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_2882_to_input_offset_3362("from_input_offset_2882_to_input_offset_3362");
#pragma HLS stream variable = from_input_offset_2882_to_input_offset_3362 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2882_to_input_offset_3362
  hls::stream<Data<int16_t>> from_input_offset_2882_to_tmp_pe_1("from_input_offset_2882_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_2882_to_tmp_pe_1 depth = 27
#pragma HLS aggregate bit variable = from_input_offset_2882_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_2883_to_input_offset_3363("from_input_offset_2883_to_input_offset_3363");
#pragma HLS stream variable = from_input_offset_2883_to_input_offset_3363 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2883_to_input_offset_3363
  hls::stream<Data<int16_t>> from_input_offset_2883_to_tmp_pe_0("from_input_offset_2883_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_2883_to_tmp_pe_0 depth = 27
#pragma HLS aggregate bit variable = from_input_offset_2883_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_3360_to_input_offset_3840("from_input_offset_3360_to_input_offset_3840");
#pragma HLS stream variable = from_input_offset_3360_to_input_offset_3840 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3360_to_input_offset_3840
  hls::stream<Data<int16_t>> from_input_offset_3360_to_tmp_pe_3("from_input_offset_3360_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_3360_to_tmp_pe_3 depth = 25
#pragma HLS aggregate bit variable = from_input_offset_3360_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_3361_to_input_offset_3841("from_input_offset_3361_to_input_offset_3841");
#pragma HLS stream variable = from_input_offset_3361_to_input_offset_3841 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3361_to_input_offset_3841
  hls::stream<Data<int16_t>> from_input_offset_3361_to_tmp_pe_2("from_input_offset_3361_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_3361_to_tmp_pe_2 depth = 25
#pragma HLS aggregate bit variable = from_input_offset_3361_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_3362_to_input_offset_3842("from_input_offset_3362_to_input_offset_3842");
#pragma HLS stream variable = from_input_offset_3362_to_input_offset_3842 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3362_to_input_offset_3842
  hls::stream<Data<int16_t>> from_input_offset_3362_to_tmp_pe_1("from_input_offset_3362_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_3362_to_tmp_pe_1 depth = 25
#pragma HLS aggregate bit variable = from_input_offset_3362_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_3363_to_input_offset_3843("from_input_offset_3363_to_input_offset_3843");
#pragma HLS stream variable = from_input_offset_3363_to_input_offset_3843 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3363_to_input_offset_3843
  hls::stream<Data<int16_t>> from_input_offset_3363_to_tmp_pe_0("from_input_offset_3363_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_3363_to_tmp_pe_0 depth = 25
#pragma HLS aggregate bit variable = from_input_offset_3363_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_3840_to_input_offset_4320("from_input_offset_3840_to_input_offset_4320");
#pragma HLS stream variable = from_input_offset_3840_to_input_offset_4320 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3840_to_input_offset_4320
  hls::stream<Data<int16_t>> from_input_offset_3840_to_tmp_pe_3("from_input_offset_3840_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_3840_to_tmp_pe_3 depth = 23
#pragma HLS aggregate bit variable = from_input_offset_3840_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_3841_to_input_offset_4321("from_input_offset_3841_to_input_offset_4321");
#pragma HLS stream variable = from_input_offset_3841_to_input_offset_4321 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3841_to_input_offset_4321
  hls::stream<Data<int16_t>> from_input_offset_3841_to_tmp_pe_2("from_input_offset_3841_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_3841_to_tmp_pe_2 depth = 23
#pragma HLS aggregate bit variable = from_input_offset_3841_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_3842_to_input_offset_4322("from_input_offset_3842_to_input_offset_4322");
#pragma HLS stream variable = from_input_offset_3842_to_input_offset_4322 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3842_to_input_offset_4322
  hls::stream<Data<int16_t>> from_input_offset_3842_to_tmp_pe_1("from_input_offset_3842_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_3842_to_tmp_pe_1 depth = 23
#pragma HLS aggregate bit variable = from_input_offset_3842_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_3843_to_input_offset_4323("from_input_offset_3843_to_input_offset_4323");
#pragma HLS stream variable = from_input_offset_3843_to_input_offset_4323 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3843_to_input_offset_4323
  hls::stream<Data<int16_t>> from_input_offset_3843_to_tmp_pe_0("from_input_offset_3843_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_3843_to_tmp_pe_0 depth = 23
#pragma HLS aggregate bit variable = from_input_offset_3843_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_4320_to_input_offset_4800("from_input_offset_4320_to_input_offset_4800");
#pragma HLS stream variable = from_input_offset_4320_to_input_offset_4800 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4320_to_input_offset_4800
  hls::stream<Data<int16_t>> from_input_offset_4320_to_tmp_pe_3("from_input_offset_4320_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_4320_to_tmp_pe_3 depth = 21
#pragma HLS aggregate bit variable = from_input_offset_4320_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_4321_to_input_offset_4801("from_input_offset_4321_to_input_offset_4801");
#pragma HLS stream variable = from_input_offset_4321_to_input_offset_4801 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4321_to_input_offset_4801
  hls::stream<Data<int16_t>> from_input_offset_4321_to_tmp_pe_2("from_input_offset_4321_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_4321_to_tmp_pe_2 depth = 21
#pragma HLS aggregate bit variable = from_input_offset_4321_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_4322_to_input_offset_4802("from_input_offset_4322_to_input_offset_4802");
#pragma HLS stream variable = from_input_offset_4322_to_input_offset_4802 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4322_to_input_offset_4802
  hls::stream<Data<int16_t>> from_input_offset_4322_to_tmp_pe_1("from_input_offset_4322_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_4322_to_tmp_pe_1 depth = 21
#pragma HLS aggregate bit variable = from_input_offset_4322_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_4323_to_input_offset_4803("from_input_offset_4323_to_input_offset_4803");
#pragma HLS stream variable = from_input_offset_4323_to_input_offset_4803 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4323_to_input_offset_4803
  hls::stream<Data<int16_t>> from_input_offset_4323_to_tmp_pe_0("from_input_offset_4323_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_4323_to_tmp_pe_0 depth = 21
#pragma HLS aggregate bit variable = from_input_offset_4323_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_4800_to_input_offset_5280("from_input_offset_4800_to_input_offset_5280");
#pragma HLS stream variable = from_input_offset_4800_to_input_offset_5280 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4800_to_input_offset_5280
  hls::stream<Data<int16_t>> from_input_offset_4800_to_tmp_pe_3("from_input_offset_4800_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_4800_to_tmp_pe_3 depth = 19
#pragma HLS aggregate bit variable = from_input_offset_4800_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_4801_to_input_offset_5281("from_input_offset_4801_to_input_offset_5281");
#pragma HLS stream variable = from_input_offset_4801_to_input_offset_5281 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4801_to_input_offset_5281
  hls::stream<Data<int16_t>> from_input_offset_4801_to_tmp_pe_2("from_input_offset_4801_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_4801_to_tmp_pe_2 depth = 19
#pragma HLS aggregate bit variable = from_input_offset_4801_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_4802_to_input_offset_5282("from_input_offset_4802_to_input_offset_5282");
#pragma HLS stream variable = from_input_offset_4802_to_input_offset_5282 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4802_to_input_offset_5282
  hls::stream<Data<int16_t>> from_input_offset_4802_to_tmp_pe_1("from_input_offset_4802_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_4802_to_tmp_pe_1 depth = 19
#pragma HLS aggregate bit variable = from_input_offset_4802_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_4803_to_input_offset_5283("from_input_offset_4803_to_input_offset_5283");
#pragma HLS stream variable = from_input_offset_4803_to_input_offset_5283 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4803_to_input_offset_5283
  hls::stream<Data<int16_t>> from_input_offset_4803_to_tmp_pe_0("from_input_offset_4803_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_4803_to_tmp_pe_0 depth = 19
#pragma HLS aggregate bit variable = from_input_offset_4803_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_5280_to_input_offset_5760("from_input_offset_5280_to_input_offset_5760");
#pragma HLS stream variable = from_input_offset_5280_to_input_offset_5760 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5280_to_input_offset_5760
  hls::stream<Data<int16_t>> from_input_offset_5280_to_tmp_pe_3("from_input_offset_5280_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_5280_to_tmp_pe_3 depth = 17
#pragma HLS aggregate bit variable = from_input_offset_5280_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_5281_to_input_offset_5761("from_input_offset_5281_to_input_offset_5761");
#pragma HLS stream variable = from_input_offset_5281_to_input_offset_5761 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5281_to_input_offset_5761
  hls::stream<Data<int16_t>> from_input_offset_5281_to_tmp_pe_2("from_input_offset_5281_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_5281_to_tmp_pe_2 depth = 17
#pragma HLS aggregate bit variable = from_input_offset_5281_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_5282_to_input_offset_5762("from_input_offset_5282_to_input_offset_5762");
#pragma HLS stream variable = from_input_offset_5282_to_input_offset_5762 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5282_to_input_offset_5762
  hls::stream<Data<int16_t>> from_input_offset_5282_to_tmp_pe_1("from_input_offset_5282_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_5282_to_tmp_pe_1 depth = 17
#pragma HLS aggregate bit variable = from_input_offset_5282_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_5283_to_input_offset_5763("from_input_offset_5283_to_input_offset_5763");
#pragma HLS stream variable = from_input_offset_5283_to_input_offset_5763 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5283_to_input_offset_5763
  hls::stream<Data<int16_t>> from_input_offset_5283_to_tmp_pe_0("from_input_offset_5283_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_5283_to_tmp_pe_0 depth = 17
#pragma HLS aggregate bit variable = from_input_offset_5283_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_5760_to_input_offset_6240("from_input_offset_5760_to_input_offset_6240");
#pragma HLS stream variable = from_input_offset_5760_to_input_offset_6240 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5760_to_input_offset_6240
  hls::stream<Data<int16_t>> from_input_offset_5760_to_tmp_pe_3("from_input_offset_5760_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_5760_to_tmp_pe_3 depth = 15
#pragma HLS aggregate bit variable = from_input_offset_5760_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_5761_to_input_offset_6241("from_input_offset_5761_to_input_offset_6241");
#pragma HLS stream variable = from_input_offset_5761_to_input_offset_6241 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5761_to_input_offset_6241
  hls::stream<Data<int16_t>> from_input_offset_5761_to_tmp_pe_2("from_input_offset_5761_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_5761_to_tmp_pe_2 depth = 15
#pragma HLS aggregate bit variable = from_input_offset_5761_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_5762_to_input_offset_6242("from_input_offset_5762_to_input_offset_6242");
#pragma HLS stream variable = from_input_offset_5762_to_input_offset_6242 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5762_to_input_offset_6242
  hls::stream<Data<int16_t>> from_input_offset_5762_to_tmp_pe_1("from_input_offset_5762_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_5762_to_tmp_pe_1 depth = 15
#pragma HLS aggregate bit variable = from_input_offset_5762_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_5763_to_input_offset_6243("from_input_offset_5763_to_input_offset_6243");
#pragma HLS stream variable = from_input_offset_5763_to_input_offset_6243 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5763_to_input_offset_6243
  hls::stream<Data<int16_t>> from_input_offset_5763_to_tmp_pe_0("from_input_offset_5763_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_5763_to_tmp_pe_0 depth = 15
#pragma HLS aggregate bit variable = from_input_offset_5763_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_6240_to_input_offset_6720("from_input_offset_6240_to_input_offset_6720");
#pragma HLS stream variable = from_input_offset_6240_to_input_offset_6720 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6240_to_input_offset_6720
  hls::stream<Data<int16_t>> from_input_offset_6240_to_tmp_pe_3("from_input_offset_6240_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_6240_to_tmp_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_input_offset_6240_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_6241_to_input_offset_6721("from_input_offset_6241_to_input_offset_6721");
#pragma HLS stream variable = from_input_offset_6241_to_input_offset_6721 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6241_to_input_offset_6721
  hls::stream<Data<int16_t>> from_input_offset_6241_to_tmp_pe_2("from_input_offset_6241_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_6241_to_tmp_pe_2 depth = 13
#pragma HLS aggregate bit variable = from_input_offset_6241_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_6242_to_input_offset_6722("from_input_offset_6242_to_input_offset_6722");
#pragma HLS stream variable = from_input_offset_6242_to_input_offset_6722 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6242_to_input_offset_6722
  hls::stream<Data<int16_t>> from_input_offset_6242_to_tmp_pe_1("from_input_offset_6242_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_6242_to_tmp_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_input_offset_6242_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_6243_to_input_offset_6723("from_input_offset_6243_to_input_offset_6723");
#pragma HLS stream variable = from_input_offset_6243_to_input_offset_6723 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6243_to_input_offset_6723
  hls::stream<Data<int16_t>> from_input_offset_6243_to_tmp_pe_0("from_input_offset_6243_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_6243_to_tmp_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_input_offset_6243_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_6720_to_input_offset_7200("from_input_offset_6720_to_input_offset_7200");
#pragma HLS stream variable = from_input_offset_6720_to_input_offset_7200 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6720_to_input_offset_7200
  hls::stream<Data<int16_t>> from_input_offset_6720_to_tmp_pe_3("from_input_offset_6720_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_6720_to_tmp_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_input_offset_6720_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_6721_to_input_offset_7201("from_input_offset_6721_to_input_offset_7201");
#pragma HLS stream variable = from_input_offset_6721_to_input_offset_7201 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6721_to_input_offset_7201
  hls::stream<Data<int16_t>> from_input_offset_6721_to_tmp_pe_2("from_input_offset_6721_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_6721_to_tmp_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_input_offset_6721_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_6722_to_input_offset_7202("from_input_offset_6722_to_input_offset_7202");
#pragma HLS stream variable = from_input_offset_6722_to_input_offset_7202 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6722_to_input_offset_7202
  hls::stream<Data<int16_t>> from_input_offset_6722_to_tmp_pe_1("from_input_offset_6722_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_6722_to_tmp_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_input_offset_6722_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_6723_to_input_offset_7203("from_input_offset_6723_to_input_offset_7203");
#pragma HLS stream variable = from_input_offset_6723_to_input_offset_7203 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6723_to_input_offset_7203
  hls::stream<Data<int16_t>> from_input_offset_6723_to_tmp_pe_0("from_input_offset_6723_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_6723_to_tmp_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_input_offset_6723_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_7200_to_input_offset_7680("from_input_offset_7200_to_input_offset_7680");
#pragma HLS stream variable = from_input_offset_7200_to_input_offset_7680 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7200_to_input_offset_7680
  hls::stream<Data<int16_t>> from_input_offset_7200_to_tmp_pe_3("from_input_offset_7200_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_7200_to_tmp_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_input_offset_7200_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_7201_to_input_offset_7681("from_input_offset_7201_to_input_offset_7681");
#pragma HLS stream variable = from_input_offset_7201_to_input_offset_7681 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7201_to_input_offset_7681
  hls::stream<Data<int16_t>> from_input_offset_7201_to_tmp_pe_2("from_input_offset_7201_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_7201_to_tmp_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_input_offset_7201_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_7202_to_input_offset_7682("from_input_offset_7202_to_input_offset_7682");
#pragma HLS stream variable = from_input_offset_7202_to_input_offset_7682 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7202_to_input_offset_7682
  hls::stream<Data<int16_t>> from_input_offset_7202_to_tmp_pe_1("from_input_offset_7202_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_7202_to_tmp_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_input_offset_7202_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_7203_to_input_offset_7683("from_input_offset_7203_to_input_offset_7683");
#pragma HLS stream variable = from_input_offset_7203_to_input_offset_7683 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7203_to_input_offset_7683
  hls::stream<Data<int16_t>> from_input_offset_7203_to_tmp_pe_0("from_input_offset_7203_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_7203_to_tmp_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_input_offset_7203_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_7680_to_input_offset_8160("from_input_offset_7680_to_input_offset_8160");
#pragma HLS stream variable = from_input_offset_7680_to_input_offset_8160 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7680_to_input_offset_8160
  hls::stream<Data<int16_t>> from_input_offset_7680_to_tmp_pe_3("from_input_offset_7680_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_7680_to_tmp_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_7680_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_7681_to_input_offset_8161("from_input_offset_7681_to_input_offset_8161");
#pragma HLS stream variable = from_input_offset_7681_to_input_offset_8161 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7681_to_input_offset_8161
  hls::stream<Data<int16_t>> from_input_offset_7681_to_tmp_pe_2("from_input_offset_7681_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_7681_to_tmp_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_7681_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_7682_to_input_offset_8162("from_input_offset_7682_to_input_offset_8162");
#pragma HLS stream variable = from_input_offset_7682_to_input_offset_8162 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7682_to_input_offset_8162
  hls::stream<Data<int16_t>> from_input_offset_7682_to_tmp_pe_1("from_input_offset_7682_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_7682_to_tmp_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_7682_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_7683_to_input_offset_8163("from_input_offset_7683_to_input_offset_8163");
#pragma HLS stream variable = from_input_offset_7683_to_input_offset_8163 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7683_to_input_offset_8163
  hls::stream<Data<int16_t>> from_input_offset_7683_to_tmp_pe_0("from_input_offset_7683_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_7683_to_tmp_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_7683_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_8160_to_input_offset_8640("from_input_offset_8160_to_input_offset_8640");
#pragma HLS stream variable = from_input_offset_8160_to_input_offset_8640 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8160_to_input_offset_8640
  hls::stream<Data<int16_t>> from_input_offset_8160_to_tmp_pe_3("from_input_offset_8160_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_8160_to_tmp_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_8160_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_input_offset_8161_to_input_offset_8641("from_input_offset_8161_to_input_offset_8641");
#pragma HLS stream variable = from_input_offset_8161_to_input_offset_8641 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8161_to_input_offset_8641
  hls::stream<Data<int16_t>> from_input_offset_8161_to_tmp_pe_2("from_input_offset_8161_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_8161_to_tmp_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_8161_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_input_offset_8162_to_input_offset_8642("from_input_offset_8162_to_input_offset_8642");
#pragma HLS stream variable = from_input_offset_8162_to_input_offset_8642 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8162_to_input_offset_8642
  hls::stream<Data<int16_t>> from_input_offset_8162_to_tmp_pe_1("from_input_offset_8162_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_8162_to_tmp_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_8162_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_input_offset_8163_to_input_offset_8643("from_input_offset_8163_to_input_offset_8643");
#pragma HLS stream variable = from_input_offset_8163_to_input_offset_8643 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8163_to_input_offset_8643
  hls::stream<Data<int16_t>> from_input_offset_8163_to_tmp_pe_0("from_input_offset_8163_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_8163_to_tmp_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_8163_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_input_offset_8640_to_tmp_pe_3("from_input_offset_8640_to_tmp_pe_3");
#pragma HLS stream variable = from_input_offset_8640_to_tmp_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8640_to_tmp_pe_3
  hls::stream<Data<int16_t>> from_tmp_pe_3_to_tmp_offset_0("from_tmp_pe_3_to_tmp_offset_0");
#pragma HLS stream variable = from_tmp_pe_3_to_tmp_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_tmp_pe_3_to_tmp_offset_0
  hls::stream<Data<int16_t>> from_tmp_offset_0_to_tmp_offset_4("from_tmp_offset_0_to_tmp_offset_4");
#pragma HLS stream variable = from_tmp_offset_0_to_tmp_offset_4 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_0_to_tmp_offset_4
  hls::stream<Data<int16_t>> from_tmp_offset_0_to_output_pe_3("from_tmp_offset_0_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_0_to_output_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_0_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_4_to_tmp_offset_8("from_tmp_offset_4_to_tmp_offset_8");
#pragma HLS stream variable = from_tmp_offset_4_to_tmp_offset_8 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_4_to_tmp_offset_8
  hls::stream<Data<int16_t>> from_tmp_offset_4_to_output_pe_3("from_tmp_offset_4_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_4_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_4_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_4_to_output_pe_2("from_tmp_offset_4_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_4_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_4_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_4_to_output_pe_1("from_tmp_offset_4_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_4_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_4_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_4_to_output_pe_0("from_tmp_offset_4_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_4_to_output_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_4_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_8_to_tmp_offset_12("from_tmp_offset_8_to_tmp_offset_12");
#pragma HLS stream variable = from_tmp_offset_8_to_tmp_offset_12 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_8_to_tmp_offset_12
  hls::stream<Data<int16_t>> from_tmp_offset_8_to_output_pe_3("from_tmp_offset_8_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_8_to_output_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_8_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_8_to_output_pe_2("from_tmp_offset_8_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_8_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_8_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_8_to_output_pe_1("from_tmp_offset_8_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_8_to_output_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_8_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_8_to_output_pe_0("from_tmp_offset_8_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_8_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_8_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_12_to_tmp_offset_16("from_tmp_offset_12_to_tmp_offset_16");
#pragma HLS stream variable = from_tmp_offset_12_to_tmp_offset_16 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_12_to_tmp_offset_16
  hls::stream<Data<int16_t>> from_tmp_offset_12_to_output_pe_3("from_tmp_offset_12_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_12_to_output_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_12_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_12_to_output_pe_2("from_tmp_offset_12_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_12_to_output_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_12_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_12_to_output_pe_1("from_tmp_offset_12_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_12_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_12_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_12_to_output_pe_0("from_tmp_offset_12_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_12_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_12_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_16_to_tmp_offset_20("from_tmp_offset_16_to_tmp_offset_20");
#pragma HLS stream variable = from_tmp_offset_16_to_tmp_offset_20 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_16_to_tmp_offset_20
  hls::stream<Data<int16_t>> from_tmp_offset_16_to_output_pe_3("from_tmp_offset_16_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_16_to_output_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_16_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_16_to_output_pe_2("from_tmp_offset_16_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_16_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_16_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_16_to_output_pe_1("from_tmp_offset_16_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_16_to_output_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_16_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_16_to_output_pe_0("from_tmp_offset_16_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_16_to_output_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_16_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_20_to_output_pe_1("from_tmp_offset_20_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_20_to_output_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_20_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_20_to_output_pe_0("from_tmp_offset_20_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_20_to_output_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_20_to_output_pe_0
  hls::stream<Data<int16_t>> from_input_offset_8641_to_tmp_pe_2("from_input_offset_8641_to_tmp_pe_2");
#pragma HLS stream variable = from_input_offset_8641_to_tmp_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8641_to_tmp_pe_2
  hls::stream<Data<int16_t>> from_tmp_pe_2_to_tmp_offset_1("from_tmp_pe_2_to_tmp_offset_1");
#pragma HLS stream variable = from_tmp_pe_2_to_tmp_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_tmp_pe_2_to_tmp_offset_1
  hls::stream<Data<int16_t>> from_tmp_offset_1_to_tmp_offset_5("from_tmp_offset_1_to_tmp_offset_5");
#pragma HLS stream variable = from_tmp_offset_1_to_tmp_offset_5 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_1_to_tmp_offset_5
  hls::stream<Data<int16_t>> from_tmp_offset_1_to_output_pe_3("from_tmp_offset_1_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_1_to_output_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_1_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_1_to_output_pe_2("from_tmp_offset_1_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_1_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_1_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_5_to_tmp_offset_9("from_tmp_offset_5_to_tmp_offset_9");
#pragma HLS stream variable = from_tmp_offset_5_to_tmp_offset_9 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_5_to_tmp_offset_9
  hls::stream<Data<int16_t>> from_tmp_offset_5_to_output_pe_3("from_tmp_offset_5_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_5_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_5_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_5_to_output_pe_2("from_tmp_offset_5_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_5_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_5_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_5_to_output_pe_1("from_tmp_offset_5_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_5_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_5_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_5_to_output_pe_0("from_tmp_offset_5_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_5_to_output_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_5_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_9_to_tmp_offset_13("from_tmp_offset_9_to_tmp_offset_13");
#pragma HLS stream variable = from_tmp_offset_9_to_tmp_offset_13 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_9_to_tmp_offset_13
  hls::stream<Data<int16_t>> from_tmp_offset_9_to_output_pe_3("from_tmp_offset_9_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_9_to_output_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_9_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_9_to_output_pe_2("from_tmp_offset_9_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_9_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_9_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_9_to_output_pe_1("from_tmp_offset_9_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_9_to_output_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_9_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_9_to_output_pe_0("from_tmp_offset_9_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_9_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_9_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_13_to_tmp_offset_17("from_tmp_offset_13_to_tmp_offset_17");
#pragma HLS stream variable = from_tmp_offset_13_to_tmp_offset_17 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_13_to_tmp_offset_17
  hls::stream<Data<int16_t>> from_tmp_offset_13_to_output_pe_3("from_tmp_offset_13_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_13_to_output_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_13_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_13_to_output_pe_2("from_tmp_offset_13_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_13_to_output_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_13_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_13_to_output_pe_1("from_tmp_offset_13_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_13_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_13_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_13_to_output_pe_0("from_tmp_offset_13_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_13_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_13_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_17_to_tmp_offset_21("from_tmp_offset_17_to_tmp_offset_21");
#pragma HLS stream variable = from_tmp_offset_17_to_tmp_offset_21 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_17_to_tmp_offset_21
  hls::stream<Data<int16_t>> from_tmp_offset_17_to_output_pe_3("from_tmp_offset_17_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_17_to_output_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_17_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_17_to_output_pe_2("from_tmp_offset_17_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_17_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_17_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_17_to_output_pe_1("from_tmp_offset_17_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_17_to_output_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_17_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_17_to_output_pe_0("from_tmp_offset_17_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_17_to_output_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_17_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_21_to_output_pe_0("from_tmp_offset_21_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_21_to_output_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_21_to_output_pe_0
  hls::stream<Data<int16_t>> from_input_offset_8642_to_tmp_pe_1("from_input_offset_8642_to_tmp_pe_1");
#pragma HLS stream variable = from_input_offset_8642_to_tmp_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8642_to_tmp_pe_1
  hls::stream<Data<int16_t>> from_tmp_pe_1_to_tmp_offset_2("from_tmp_pe_1_to_tmp_offset_2");
#pragma HLS stream variable = from_tmp_pe_1_to_tmp_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_tmp_pe_1_to_tmp_offset_2
  hls::stream<Data<int16_t>> from_tmp_offset_2_to_tmp_offset_6("from_tmp_offset_2_to_tmp_offset_6");
#pragma HLS stream variable = from_tmp_offset_2_to_tmp_offset_6 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_2_to_tmp_offset_6
  hls::stream<Data<int16_t>> from_tmp_offset_2_to_output_pe_3("from_tmp_offset_2_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_2_to_output_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_2_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_2_to_output_pe_2("from_tmp_offset_2_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_2_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_2_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_2_to_output_pe_1("from_tmp_offset_2_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_2_to_output_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_tmp_offset_2_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_6_to_tmp_offset_10("from_tmp_offset_6_to_tmp_offset_10");
#pragma HLS stream variable = from_tmp_offset_6_to_tmp_offset_10 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_6_to_tmp_offset_10
  hls::stream<Data<int16_t>> from_tmp_offset_6_to_output_pe_3("from_tmp_offset_6_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_6_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_6_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_6_to_output_pe_2("from_tmp_offset_6_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_6_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_6_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_6_to_output_pe_1("from_tmp_offset_6_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_6_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_6_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_6_to_output_pe_0("from_tmp_offset_6_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_6_to_output_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_6_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_10_to_tmp_offset_14("from_tmp_offset_10_to_tmp_offset_14");
#pragma HLS stream variable = from_tmp_offset_10_to_tmp_offset_14 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_10_to_tmp_offset_14
  hls::stream<Data<int16_t>> from_tmp_offset_10_to_output_pe_3("from_tmp_offset_10_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_10_to_output_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_10_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_10_to_output_pe_2("from_tmp_offset_10_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_10_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_10_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_10_to_output_pe_1("from_tmp_offset_10_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_10_to_output_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_10_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_10_to_output_pe_0("from_tmp_offset_10_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_10_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_10_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_14_to_tmp_offset_18("from_tmp_offset_14_to_tmp_offset_18");
#pragma HLS stream variable = from_tmp_offset_14_to_tmp_offset_18 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_14_to_tmp_offset_18
  hls::stream<Data<int16_t>> from_tmp_offset_14_to_output_pe_3("from_tmp_offset_14_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_14_to_output_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_14_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_14_to_output_pe_2("from_tmp_offset_14_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_14_to_output_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_14_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_14_to_output_pe_1("from_tmp_offset_14_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_14_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_14_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_14_to_output_pe_0("from_tmp_offset_14_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_14_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_14_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_18_to_output_pe_3("from_tmp_offset_18_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_18_to_output_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_18_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_18_to_output_pe_2("from_tmp_offset_18_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_18_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_18_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_18_to_output_pe_1("from_tmp_offset_18_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_18_to_output_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_18_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_18_to_output_pe_0("from_tmp_offset_18_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_18_to_output_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_18_to_output_pe_0
  hls::stream<Data<int16_t>> from_input_offset_8643_to_tmp_pe_0("from_input_offset_8643_to_tmp_pe_0");
#pragma HLS stream variable = from_input_offset_8643_to_tmp_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8643_to_tmp_pe_0
  hls::stream<Data<int16_t>> from_tmp_pe_0_to_tmp_offset_3("from_tmp_pe_0_to_tmp_offset_3");
#pragma HLS stream variable = from_tmp_pe_0_to_tmp_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_tmp_pe_0_to_tmp_offset_3
  hls::stream<Data<int16_t>> from_tmp_offset_3_to_tmp_offset_7("from_tmp_offset_3_to_tmp_offset_7");
#pragma HLS stream variable = from_tmp_offset_3_to_tmp_offset_7 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_3_to_tmp_offset_7
  hls::stream<Data<int16_t>> from_tmp_offset_3_to_output_pe_3("from_tmp_offset_3_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_3_to_output_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_3_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_3_to_output_pe_2("from_tmp_offset_3_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_3_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_3_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_3_to_output_pe_1("from_tmp_offset_3_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_3_to_output_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_tmp_offset_3_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_3_to_output_pe_0("from_tmp_offset_3_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_3_to_output_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_tmp_offset_3_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_7_to_tmp_offset_11("from_tmp_offset_7_to_tmp_offset_11");
#pragma HLS stream variable = from_tmp_offset_7_to_tmp_offset_11 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_7_to_tmp_offset_11
  hls::stream<Data<int16_t>> from_tmp_offset_7_to_output_pe_3("from_tmp_offset_7_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_7_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_7_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_7_to_output_pe_2("from_tmp_offset_7_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_7_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_7_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_7_to_output_pe_1("from_tmp_offset_7_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_7_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_7_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_7_to_output_pe_0("from_tmp_offset_7_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_7_to_output_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_tmp_offset_7_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_11_to_tmp_offset_15("from_tmp_offset_11_to_tmp_offset_15");
#pragma HLS stream variable = from_tmp_offset_11_to_tmp_offset_15 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_11_to_tmp_offset_15
  hls::stream<Data<int16_t>> from_tmp_offset_11_to_output_pe_3("from_tmp_offset_11_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_11_to_output_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_11_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_11_to_output_pe_2("from_tmp_offset_11_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_11_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_11_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_11_to_output_pe_1("from_tmp_offset_11_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_11_to_output_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_11_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_11_to_output_pe_0("from_tmp_offset_11_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_11_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_tmp_offset_11_to_output_pe_0
  hls::stream<Data<int16_t>> from_tmp_offset_15_to_tmp_offset_19("from_tmp_offset_15_to_tmp_offset_19");
#pragma HLS stream variable = from_tmp_offset_15_to_tmp_offset_19 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_15_to_tmp_offset_19
  hls::stream<Data<int16_t>> from_tmp_offset_15_to_output_pe_3("from_tmp_offset_15_to_output_pe_3");
#pragma HLS stream variable = from_tmp_offset_15_to_output_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_15_to_output_pe_3
  hls::stream<Data<int16_t>> from_tmp_offset_15_to_output_pe_2("from_tmp_offset_15_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_15_to_output_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_15_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_15_to_output_pe_1("from_tmp_offset_15_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_15_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_15_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_15_to_output_pe_0("from_tmp_offset_15_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_15_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_tmp_offset_15_to_output_pe_0
  hls::stream<Data<int16_t>> from_output_pe_3_to_output_bank_1("from_output_pe_3_to_output_bank_1");
#pragma HLS stream variable = from_output_pe_3_to_output_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_output_pe_3_to_output_bank_1
  hls::stream<Data<int16_t>> from_tmp_offset_19_to_output_pe_2("from_tmp_offset_19_to_output_pe_2");
#pragma HLS stream variable = from_tmp_offset_19_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_tmp_offset_19_to_output_pe_2
  hls::stream<Data<int16_t>> from_tmp_offset_19_to_output_pe_1("from_tmp_offset_19_to_output_pe_1");
#pragma HLS stream variable = from_tmp_offset_19_to_output_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_19_to_output_pe_1
  hls::stream<Data<int16_t>> from_tmp_offset_19_to_output_pe_0("from_tmp_offset_19_to_output_pe_0");
#pragma HLS stream variable = from_tmp_offset_19_to_output_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_tmp_offset_19_to_output_pe_0
  hls::stream<Data<int16_t>> from_output_pe_2_to_output_bank_1("from_output_pe_2_to_output_bank_1");
#pragma HLS stream variable = from_output_pe_2_to_output_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_output_pe_2_to_output_bank_1
  hls::stream<Data<int16_t>> from_output_pe_1_to_output_bank_1("from_output_pe_1_to_output_bank_1");
#pragma HLS stream variable = from_output_pe_1_to_output_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_output_pe_1_to_output_bank_1
  hls::stream<Data<int16_t>> from_output_pe_0_to_output_bank_1("from_output_pe_0_to_output_bank_1");
#pragma HLS stream variable = from_output_pe_0_to_output_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_output_pe_0_to_output_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_input_buf, bank_0_input, coalesced_data_num);
  Module0Func(
      /*output*/ from_input_bank_0_to_input_offset_0, 
      /*output*/ from_input_bank_0_to_input_offset_1, 
      /*output*/ from_input_bank_0_to_input_offset_2, 
      /*output*/ from_input_bank_0_to_input_offset_3, 
      /* input*/ bank_0_input_buf);
  Module1Func(
      /*output*/ from_input_offset_0_to_input_offset_480, 
      /*output*/ from_input_offset_0_to_tmp_pe_3, 
      /* input*/ from_input_bank_0_to_input_offset_0);
  Module1Func(
      /*output*/ from_input_offset_1_to_input_offset_481, 
      /*output*/ from_input_offset_1_to_tmp_pe_2, 
      /* input*/ from_input_bank_0_to_input_offset_1);
  Module1Func(
      /*output*/ from_input_offset_2_to_input_offset_482, 
      /*output*/ from_input_offset_2_to_tmp_pe_1, 
      /* input*/ from_input_bank_0_to_input_offset_2);
  Module1Func(
      /*output*/ from_input_offset_3_to_input_offset_483, 
      /*output*/ from_input_offset_3_to_tmp_pe_0, 
      /* input*/ from_input_bank_0_to_input_offset_3);
  Module2Func(
      /*output*/ from_input_offset_480_to_input_offset_960, 
      /*output*/ from_input_offset_480_to_tmp_pe_3, 
      /* input*/ from_input_offset_0_to_input_offset_480);
  Module2Func(
      /*output*/ from_input_offset_481_to_input_offset_961, 
      /*output*/ from_input_offset_481_to_tmp_pe_2, 
      /* input*/ from_input_offset_1_to_input_offset_481);
  Module2Func(
      /*output*/ from_input_offset_482_to_input_offset_962, 
      /*output*/ from_input_offset_482_to_tmp_pe_1, 
      /* input*/ from_input_offset_2_to_input_offset_482);
  Module2Func(
      /*output*/ from_input_offset_483_to_input_offset_963, 
      /*output*/ from_input_offset_483_to_tmp_pe_0, 
      /* input*/ from_input_offset_3_to_input_offset_483);
  Module2Func(
      /*output*/ from_input_offset_960_to_input_offset_1440, 
      /*output*/ from_input_offset_960_to_tmp_pe_3, 
      /* input*/ from_input_offset_480_to_input_offset_960);
  Module2Func(
      /*output*/ from_input_offset_961_to_input_offset_1441, 
      /*output*/ from_input_offset_961_to_tmp_pe_2, 
      /* input*/ from_input_offset_481_to_input_offset_961);
  Module2Func(
      /*output*/ from_input_offset_962_to_input_offset_1442, 
      /*output*/ from_input_offset_962_to_tmp_pe_1, 
      /* input*/ from_input_offset_482_to_input_offset_962);
  Module2Func(
      /*output*/ from_input_offset_963_to_input_offset_1443, 
      /*output*/ from_input_offset_963_to_tmp_pe_0, 
      /* input*/ from_input_offset_483_to_input_offset_963);
  Module2Func(
      /*output*/ from_input_offset_1440_to_input_offset_1920, 
      /*output*/ from_input_offset_1440_to_tmp_pe_3, 
      /* input*/ from_input_offset_960_to_input_offset_1440);
  Module2Func(
      /*output*/ from_input_offset_1441_to_input_offset_1921, 
      /*output*/ from_input_offset_1441_to_tmp_pe_2, 
      /* input*/ from_input_offset_961_to_input_offset_1441);
  Module2Func(
      /*output*/ from_input_offset_1442_to_input_offset_1922, 
      /*output*/ from_input_offset_1442_to_tmp_pe_1, 
      /* input*/ from_input_offset_962_to_input_offset_1442);
  Module2Func(
      /*output*/ from_input_offset_1443_to_input_offset_1923, 
      /*output*/ from_input_offset_1443_to_tmp_pe_0, 
      /* input*/ from_input_offset_963_to_input_offset_1443);
  Module2Func(
      /*output*/ from_input_offset_1920_to_input_offset_2400, 
      /*output*/ from_input_offset_1920_to_tmp_pe_3, 
      /* input*/ from_input_offset_1440_to_input_offset_1920);
  Module2Func(
      /*output*/ from_input_offset_1921_to_input_offset_2401, 
      /*output*/ from_input_offset_1921_to_tmp_pe_2, 
      /* input*/ from_input_offset_1441_to_input_offset_1921);
  Module2Func(
      /*output*/ from_input_offset_1922_to_input_offset_2402, 
      /*output*/ from_input_offset_1922_to_tmp_pe_1, 
      /* input*/ from_input_offset_1442_to_input_offset_1922);
  Module2Func(
      /*output*/ from_input_offset_1923_to_input_offset_2403, 
      /*output*/ from_input_offset_1923_to_tmp_pe_0, 
      /* input*/ from_input_offset_1443_to_input_offset_1923);
  Module2Func(
      /*output*/ from_input_offset_2400_to_input_offset_2880, 
      /*output*/ from_input_offset_2400_to_tmp_pe_3, 
      /* input*/ from_input_offset_1920_to_input_offset_2400);
  Module2Func(
      /*output*/ from_input_offset_2401_to_input_offset_2881, 
      /*output*/ from_input_offset_2401_to_tmp_pe_2, 
      /* input*/ from_input_offset_1921_to_input_offset_2401);
  Module2Func(
      /*output*/ from_input_offset_2402_to_input_offset_2882, 
      /*output*/ from_input_offset_2402_to_tmp_pe_1, 
      /* input*/ from_input_offset_1922_to_input_offset_2402);
  Module2Func(
      /*output*/ from_input_offset_2403_to_input_offset_2883, 
      /*output*/ from_input_offset_2403_to_tmp_pe_0, 
      /* input*/ from_input_offset_1923_to_input_offset_2403);
  Module2Func(
      /*output*/ from_input_offset_2880_to_input_offset_3360, 
      /*output*/ from_input_offset_2880_to_tmp_pe_3, 
      /* input*/ from_input_offset_2400_to_input_offset_2880);
  Module2Func(
      /*output*/ from_input_offset_2881_to_input_offset_3361, 
      /*output*/ from_input_offset_2881_to_tmp_pe_2, 
      /* input*/ from_input_offset_2401_to_input_offset_2881);
  Module2Func(
      /*output*/ from_input_offset_2882_to_input_offset_3362, 
      /*output*/ from_input_offset_2882_to_tmp_pe_1, 
      /* input*/ from_input_offset_2402_to_input_offset_2882);
  Module2Func(
      /*output*/ from_input_offset_2883_to_input_offset_3363, 
      /*output*/ from_input_offset_2883_to_tmp_pe_0, 
      /* input*/ from_input_offset_2403_to_input_offset_2883);
  Module2Func(
      /*output*/ from_input_offset_3360_to_input_offset_3840, 
      /*output*/ from_input_offset_3360_to_tmp_pe_3, 
      /* input*/ from_input_offset_2880_to_input_offset_3360);
  Module2Func(
      /*output*/ from_input_offset_3361_to_input_offset_3841, 
      /*output*/ from_input_offset_3361_to_tmp_pe_2, 
      /* input*/ from_input_offset_2881_to_input_offset_3361);
  Module2Func(
      /*output*/ from_input_offset_3362_to_input_offset_3842, 
      /*output*/ from_input_offset_3362_to_tmp_pe_1, 
      /* input*/ from_input_offset_2882_to_input_offset_3362);
  Module2Func(
      /*output*/ from_input_offset_3363_to_input_offset_3843, 
      /*output*/ from_input_offset_3363_to_tmp_pe_0, 
      /* input*/ from_input_offset_2883_to_input_offset_3363);
  Module2Func(
      /*output*/ from_input_offset_3840_to_input_offset_4320, 
      /*output*/ from_input_offset_3840_to_tmp_pe_3, 
      /* input*/ from_input_offset_3360_to_input_offset_3840);
  Module2Func(
      /*output*/ from_input_offset_3841_to_input_offset_4321, 
      /*output*/ from_input_offset_3841_to_tmp_pe_2, 
      /* input*/ from_input_offset_3361_to_input_offset_3841);
  Module2Func(
      /*output*/ from_input_offset_3842_to_input_offset_4322, 
      /*output*/ from_input_offset_3842_to_tmp_pe_1, 
      /* input*/ from_input_offset_3362_to_input_offset_3842);
  Module2Func(
      /*output*/ from_input_offset_3843_to_input_offset_4323, 
      /*output*/ from_input_offset_3843_to_tmp_pe_0, 
      /* input*/ from_input_offset_3363_to_input_offset_3843);
  Module2Func(
      /*output*/ from_input_offset_4320_to_input_offset_4800, 
      /*output*/ from_input_offset_4320_to_tmp_pe_3, 
      /* input*/ from_input_offset_3840_to_input_offset_4320);
  Module2Func(
      /*output*/ from_input_offset_4321_to_input_offset_4801, 
      /*output*/ from_input_offset_4321_to_tmp_pe_2, 
      /* input*/ from_input_offset_3841_to_input_offset_4321);
  Module2Func(
      /*output*/ from_input_offset_4322_to_input_offset_4802, 
      /*output*/ from_input_offset_4322_to_tmp_pe_1, 
      /* input*/ from_input_offset_3842_to_input_offset_4322);
  Module2Func(
      /*output*/ from_input_offset_4323_to_input_offset_4803, 
      /*output*/ from_input_offset_4323_to_tmp_pe_0, 
      /* input*/ from_input_offset_3843_to_input_offset_4323);
  Module2Func(
      /*output*/ from_input_offset_4800_to_input_offset_5280, 
      /*output*/ from_input_offset_4800_to_tmp_pe_3, 
      /* input*/ from_input_offset_4320_to_input_offset_4800);
  Module2Func(
      /*output*/ from_input_offset_4801_to_input_offset_5281, 
      /*output*/ from_input_offset_4801_to_tmp_pe_2, 
      /* input*/ from_input_offset_4321_to_input_offset_4801);
  Module2Func(
      /*output*/ from_input_offset_4802_to_input_offset_5282, 
      /*output*/ from_input_offset_4802_to_tmp_pe_1, 
      /* input*/ from_input_offset_4322_to_input_offset_4802);
  Module2Func(
      /*output*/ from_input_offset_4803_to_input_offset_5283, 
      /*output*/ from_input_offset_4803_to_tmp_pe_0, 
      /* input*/ from_input_offset_4323_to_input_offset_4803);
  Module2Func(
      /*output*/ from_input_offset_5280_to_input_offset_5760, 
      /*output*/ from_input_offset_5280_to_tmp_pe_3, 
      /* input*/ from_input_offset_4800_to_input_offset_5280);
  Module2Func(
      /*output*/ from_input_offset_5281_to_input_offset_5761, 
      /*output*/ from_input_offset_5281_to_tmp_pe_2, 
      /* input*/ from_input_offset_4801_to_input_offset_5281);
  Module2Func(
      /*output*/ from_input_offset_5282_to_input_offset_5762, 
      /*output*/ from_input_offset_5282_to_tmp_pe_1, 
      /* input*/ from_input_offset_4802_to_input_offset_5282);
  Module2Func(
      /*output*/ from_input_offset_5283_to_input_offset_5763, 
      /*output*/ from_input_offset_5283_to_tmp_pe_0, 
      /* input*/ from_input_offset_4803_to_input_offset_5283);
  Module2Func(
      /*output*/ from_input_offset_5760_to_input_offset_6240, 
      /*output*/ from_input_offset_5760_to_tmp_pe_3, 
      /* input*/ from_input_offset_5280_to_input_offset_5760);
  Module2Func(
      /*output*/ from_input_offset_5761_to_input_offset_6241, 
      /*output*/ from_input_offset_5761_to_tmp_pe_2, 
      /* input*/ from_input_offset_5281_to_input_offset_5761);
  Module2Func(
      /*output*/ from_input_offset_5762_to_input_offset_6242, 
      /*output*/ from_input_offset_5762_to_tmp_pe_1, 
      /* input*/ from_input_offset_5282_to_input_offset_5762);
  Module2Func(
      /*output*/ from_input_offset_5763_to_input_offset_6243, 
      /*output*/ from_input_offset_5763_to_tmp_pe_0, 
      /* input*/ from_input_offset_5283_to_input_offset_5763);
  Module2Func(
      /*output*/ from_input_offset_6240_to_input_offset_6720, 
      /*output*/ from_input_offset_6240_to_tmp_pe_3, 
      /* input*/ from_input_offset_5760_to_input_offset_6240);
  Module2Func(
      /*output*/ from_input_offset_6241_to_input_offset_6721, 
      /*output*/ from_input_offset_6241_to_tmp_pe_2, 
      /* input*/ from_input_offset_5761_to_input_offset_6241);
  Module2Func(
      /*output*/ from_input_offset_6242_to_input_offset_6722, 
      /*output*/ from_input_offset_6242_to_tmp_pe_1, 
      /* input*/ from_input_offset_5762_to_input_offset_6242);
  Module2Func(
      /*output*/ from_input_offset_6243_to_input_offset_6723, 
      /*output*/ from_input_offset_6243_to_tmp_pe_0, 
      /* input*/ from_input_offset_5763_to_input_offset_6243);
  Module2Func(
      /*output*/ from_input_offset_6720_to_input_offset_7200, 
      /*output*/ from_input_offset_6720_to_tmp_pe_3, 
      /* input*/ from_input_offset_6240_to_input_offset_6720);
  Module2Func(
      /*output*/ from_input_offset_6721_to_input_offset_7201, 
      /*output*/ from_input_offset_6721_to_tmp_pe_2, 
      /* input*/ from_input_offset_6241_to_input_offset_6721);
  Module2Func(
      /*output*/ from_input_offset_6722_to_input_offset_7202, 
      /*output*/ from_input_offset_6722_to_tmp_pe_1, 
      /* input*/ from_input_offset_6242_to_input_offset_6722);
  Module2Func(
      /*output*/ from_input_offset_6723_to_input_offset_7203, 
      /*output*/ from_input_offset_6723_to_tmp_pe_0, 
      /* input*/ from_input_offset_6243_to_input_offset_6723);
  Module2Func(
      /*output*/ from_input_offset_7200_to_input_offset_7680, 
      /*output*/ from_input_offset_7200_to_tmp_pe_3, 
      /* input*/ from_input_offset_6720_to_input_offset_7200);
  Module2Func(
      /*output*/ from_input_offset_7201_to_input_offset_7681, 
      /*output*/ from_input_offset_7201_to_tmp_pe_2, 
      /* input*/ from_input_offset_6721_to_input_offset_7201);
  Module2Func(
      /*output*/ from_input_offset_7202_to_input_offset_7682, 
      /*output*/ from_input_offset_7202_to_tmp_pe_1, 
      /* input*/ from_input_offset_6722_to_input_offset_7202);
  Module2Func(
      /*output*/ from_input_offset_7203_to_input_offset_7683, 
      /*output*/ from_input_offset_7203_to_tmp_pe_0, 
      /* input*/ from_input_offset_6723_to_input_offset_7203);
  Module2Func(
      /*output*/ from_input_offset_7680_to_input_offset_8160, 
      /*output*/ from_input_offset_7680_to_tmp_pe_3, 
      /* input*/ from_input_offset_7200_to_input_offset_7680);
  Module2Func(
      /*output*/ from_input_offset_7681_to_input_offset_8161, 
      /*output*/ from_input_offset_7681_to_tmp_pe_2, 
      /* input*/ from_input_offset_7201_to_input_offset_7681);
  Module2Func(
      /*output*/ from_input_offset_7682_to_input_offset_8162, 
      /*output*/ from_input_offset_7682_to_tmp_pe_1, 
      /* input*/ from_input_offset_7202_to_input_offset_7682);
  Module2Func(
      /*output*/ from_input_offset_7683_to_input_offset_8163, 
      /*output*/ from_input_offset_7683_to_tmp_pe_0, 
      /* input*/ from_input_offset_7203_to_input_offset_7683);
  Module2Func(
      /*output*/ from_input_offset_8160_to_input_offset_8640, 
      /*output*/ from_input_offset_8160_to_tmp_pe_3, 
      /* input*/ from_input_offset_7680_to_input_offset_8160);
  Module2Func(
      /*output*/ from_input_offset_8161_to_input_offset_8641, 
      /*output*/ from_input_offset_8161_to_tmp_pe_2, 
      /* input*/ from_input_offset_7681_to_input_offset_8161);
  Module2Func(
      /*output*/ from_input_offset_8162_to_input_offset_8642, 
      /*output*/ from_input_offset_8162_to_tmp_pe_1, 
      /* input*/ from_input_offset_7682_to_input_offset_8162);
  Module2Func(
      /*output*/ from_input_offset_8163_to_input_offset_8643, 
      /*output*/ from_input_offset_8163_to_tmp_pe_0, 
      /* input*/ from_input_offset_7683_to_input_offset_8163);
  Module3Func(
      /*output*/ from_input_offset_8640_to_tmp_pe_3, 
      /* input*/ from_input_offset_8160_to_input_offset_8640);
  Module4Func(
      /*output*/ from_tmp_pe_3_to_tmp_offset_0, 
      /* input*/ from_input_offset_8640_to_tmp_pe_3, 
      /* input*/ from_input_offset_8160_to_tmp_pe_3, 
      /* input*/ from_input_offset_7680_to_tmp_pe_3, 
      /* input*/ from_input_offset_7200_to_tmp_pe_3, 
      /* input*/ from_input_offset_6720_to_tmp_pe_3, 
      /* input*/ from_input_offset_6240_to_tmp_pe_3, 
      /* input*/ from_input_offset_5760_to_tmp_pe_3, 
      /* input*/ from_input_offset_5280_to_tmp_pe_3, 
      /* input*/ from_input_offset_4800_to_tmp_pe_3, 
      /* input*/ from_input_offset_4320_to_tmp_pe_3, 
      /* input*/ from_input_offset_3840_to_tmp_pe_3, 
      /* input*/ from_input_offset_3360_to_tmp_pe_3, 
      /* input*/ from_input_offset_2880_to_tmp_pe_3, 
      /* input*/ from_input_offset_2400_to_tmp_pe_3, 
      /* input*/ from_input_offset_1920_to_tmp_pe_3, 
      /* input*/ from_input_offset_1440_to_tmp_pe_3, 
      /* input*/ from_input_offset_960_to_tmp_pe_3, 
      /* input*/ from_input_offset_480_to_tmp_pe_3, 
      /* input*/ from_input_offset_0_to_tmp_pe_3);
  Module1Func(
      /*output*/ from_tmp_offset_0_to_tmp_offset_4, 
      /*output*/ from_tmp_offset_0_to_output_pe_3, 
      /* input*/ from_tmp_pe_3_to_tmp_offset_0);
  Module5Func(
      /*output*/ from_tmp_offset_4_to_tmp_offset_8, 
      /*output*/ from_tmp_offset_4_to_output_pe_3, 
      /*output*/ from_tmp_offset_4_to_output_pe_2, 
      /*output*/ from_tmp_offset_4_to_output_pe_1, 
      /*output*/ from_tmp_offset_4_to_output_pe_0, 
      /* input*/ from_tmp_offset_0_to_tmp_offset_4);
  Module5Func(
      /*output*/ from_tmp_offset_8_to_tmp_offset_12, 
      /*output*/ from_tmp_offset_8_to_output_pe_3, 
      /*output*/ from_tmp_offset_8_to_output_pe_2, 
      /*output*/ from_tmp_offset_8_to_output_pe_1, 
      /*output*/ from_tmp_offset_8_to_output_pe_0, 
      /* input*/ from_tmp_offset_4_to_tmp_offset_8);
  Module5Func(
      /*output*/ from_tmp_offset_12_to_tmp_offset_16, 
      /*output*/ from_tmp_offset_12_to_output_pe_3, 
      /*output*/ from_tmp_offset_12_to_output_pe_2, 
      /*output*/ from_tmp_offset_12_to_output_pe_1, 
      /*output*/ from_tmp_offset_12_to_output_pe_0, 
      /* input*/ from_tmp_offset_8_to_tmp_offset_12);
  Module5Func(
      /*output*/ from_tmp_offset_16_to_tmp_offset_20, 
      /*output*/ from_tmp_offset_16_to_output_pe_3, 
      /*output*/ from_tmp_offset_16_to_output_pe_2, 
      /*output*/ from_tmp_offset_16_to_output_pe_1, 
      /*output*/ from_tmp_offset_16_to_output_pe_0, 
      /* input*/ from_tmp_offset_12_to_tmp_offset_16);
  Module6Func(
      /*output*/ from_tmp_offset_20_to_output_pe_1, 
      /*output*/ from_tmp_offset_20_to_output_pe_0, 
      /* input*/ from_tmp_offset_16_to_tmp_offset_20);
  Module3Func(
      /*output*/ from_input_offset_8641_to_tmp_pe_2, 
      /* input*/ from_input_offset_8161_to_input_offset_8641);
  Module4Func(
      /*output*/ from_tmp_pe_2_to_tmp_offset_1, 
      /* input*/ from_input_offset_8641_to_tmp_pe_2, 
      /* input*/ from_input_offset_8161_to_tmp_pe_2, 
      /* input*/ from_input_offset_7681_to_tmp_pe_2, 
      /* input*/ from_input_offset_7201_to_tmp_pe_2, 
      /* input*/ from_input_offset_6721_to_tmp_pe_2, 
      /* input*/ from_input_offset_6241_to_tmp_pe_2, 
      /* input*/ from_input_offset_5761_to_tmp_pe_2, 
      /* input*/ from_input_offset_5281_to_tmp_pe_2, 
      /* input*/ from_input_offset_4801_to_tmp_pe_2, 
      /* input*/ from_input_offset_4321_to_tmp_pe_2, 
      /* input*/ from_input_offset_3841_to_tmp_pe_2, 
      /* input*/ from_input_offset_3361_to_tmp_pe_2, 
      /* input*/ from_input_offset_2881_to_tmp_pe_2, 
      /* input*/ from_input_offset_2401_to_tmp_pe_2, 
      /* input*/ from_input_offset_1921_to_tmp_pe_2, 
      /* input*/ from_input_offset_1441_to_tmp_pe_2, 
      /* input*/ from_input_offset_961_to_tmp_pe_2, 
      /* input*/ from_input_offset_481_to_tmp_pe_2, 
      /* input*/ from_input_offset_1_to_tmp_pe_2);
  Module7Func(
      /*output*/ from_tmp_offset_1_to_tmp_offset_5, 
      /*output*/ from_tmp_offset_1_to_output_pe_3, 
      /*output*/ from_tmp_offset_1_to_output_pe_2, 
      /* input*/ from_tmp_pe_2_to_tmp_offset_1);
  Module5Func(
      /*output*/ from_tmp_offset_5_to_tmp_offset_9, 
      /*output*/ from_tmp_offset_5_to_output_pe_3, 
      /*output*/ from_tmp_offset_5_to_output_pe_2, 
      /*output*/ from_tmp_offset_5_to_output_pe_1, 
      /*output*/ from_tmp_offset_5_to_output_pe_0, 
      /* input*/ from_tmp_offset_1_to_tmp_offset_5);
  Module5Func(
      /*output*/ from_tmp_offset_9_to_tmp_offset_13, 
      /*output*/ from_tmp_offset_9_to_output_pe_3, 
      /*output*/ from_tmp_offset_9_to_output_pe_2, 
      /*output*/ from_tmp_offset_9_to_output_pe_1, 
      /*output*/ from_tmp_offset_9_to_output_pe_0, 
      /* input*/ from_tmp_offset_5_to_tmp_offset_9);
  Module5Func(
      /*output*/ from_tmp_offset_13_to_tmp_offset_17, 
      /*output*/ from_tmp_offset_13_to_output_pe_3, 
      /*output*/ from_tmp_offset_13_to_output_pe_2, 
      /*output*/ from_tmp_offset_13_to_output_pe_1, 
      /*output*/ from_tmp_offset_13_to_output_pe_0, 
      /* input*/ from_tmp_offset_9_to_tmp_offset_13);
  Module5Func(
      /*output*/ from_tmp_offset_17_to_tmp_offset_21, 
      /*output*/ from_tmp_offset_17_to_output_pe_3, 
      /*output*/ from_tmp_offset_17_to_output_pe_2, 
      /*output*/ from_tmp_offset_17_to_output_pe_1, 
      /*output*/ from_tmp_offset_17_to_output_pe_0, 
      /* input*/ from_tmp_offset_13_to_tmp_offset_17);
  Module8Func(
      /*output*/ from_tmp_offset_21_to_output_pe_0, 
      /* input*/ from_tmp_offset_17_to_tmp_offset_21);
  Module3Func(
      /*output*/ from_input_offset_8642_to_tmp_pe_1, 
      /* input*/ from_input_offset_8162_to_input_offset_8642);
  Module4Func(
      /*output*/ from_tmp_pe_1_to_tmp_offset_2, 
      /* input*/ from_input_offset_8642_to_tmp_pe_1, 
      /* input*/ from_input_offset_8162_to_tmp_pe_1, 
      /* input*/ from_input_offset_7682_to_tmp_pe_1, 
      /* input*/ from_input_offset_7202_to_tmp_pe_1, 
      /* input*/ from_input_offset_6722_to_tmp_pe_1, 
      /* input*/ from_input_offset_6242_to_tmp_pe_1, 
      /* input*/ from_input_offset_5762_to_tmp_pe_1, 
      /* input*/ from_input_offset_5282_to_tmp_pe_1, 
      /* input*/ from_input_offset_4802_to_tmp_pe_1, 
      /* input*/ from_input_offset_4322_to_tmp_pe_1, 
      /* input*/ from_input_offset_3842_to_tmp_pe_1, 
      /* input*/ from_input_offset_3362_to_tmp_pe_1, 
      /* input*/ from_input_offset_2882_to_tmp_pe_1, 
      /* input*/ from_input_offset_2402_to_tmp_pe_1, 
      /* input*/ from_input_offset_1922_to_tmp_pe_1, 
      /* input*/ from_input_offset_1442_to_tmp_pe_1, 
      /* input*/ from_input_offset_962_to_tmp_pe_1, 
      /* input*/ from_input_offset_482_to_tmp_pe_1, 
      /* input*/ from_input_offset_2_to_tmp_pe_1);
  Module9Func(
      /*output*/ from_tmp_offset_2_to_tmp_offset_6, 
      /*output*/ from_tmp_offset_2_to_output_pe_3, 
      /*output*/ from_tmp_offset_2_to_output_pe_2, 
      /*output*/ from_tmp_offset_2_to_output_pe_1, 
      /* input*/ from_tmp_pe_1_to_tmp_offset_2);
  Module5Func(
      /*output*/ from_tmp_offset_6_to_tmp_offset_10, 
      /*output*/ from_tmp_offset_6_to_output_pe_3, 
      /*output*/ from_tmp_offset_6_to_output_pe_2, 
      /*output*/ from_tmp_offset_6_to_output_pe_1, 
      /*output*/ from_tmp_offset_6_to_output_pe_0, 
      /* input*/ from_tmp_offset_2_to_tmp_offset_6);
  Module5Func(
      /*output*/ from_tmp_offset_10_to_tmp_offset_14, 
      /*output*/ from_tmp_offset_10_to_output_pe_3, 
      /*output*/ from_tmp_offset_10_to_output_pe_2, 
      /*output*/ from_tmp_offset_10_to_output_pe_1, 
      /*output*/ from_tmp_offset_10_to_output_pe_0, 
      /* input*/ from_tmp_offset_6_to_tmp_offset_10);
  Module5Func(
      /*output*/ from_tmp_offset_14_to_tmp_offset_18, 
      /*output*/ from_tmp_offset_14_to_output_pe_3, 
      /*output*/ from_tmp_offset_14_to_output_pe_2, 
      /*output*/ from_tmp_offset_14_to_output_pe_1, 
      /*output*/ from_tmp_offset_14_to_output_pe_0, 
      /* input*/ from_tmp_offset_10_to_tmp_offset_14);
  Module10Func(
      /*output*/ from_tmp_offset_18_to_output_pe_3, 
      /*output*/ from_tmp_offset_18_to_output_pe_2, 
      /*output*/ from_tmp_offset_18_to_output_pe_1, 
      /*output*/ from_tmp_offset_18_to_output_pe_0, 
      /* input*/ from_tmp_offset_14_to_tmp_offset_18);
  Module3Func(
      /*output*/ from_input_offset_8643_to_tmp_pe_0, 
      /* input*/ from_input_offset_8163_to_input_offset_8643);
  Module4Func(
      /*output*/ from_tmp_pe_0_to_tmp_offset_3, 
      /* input*/ from_input_offset_8643_to_tmp_pe_0, 
      /* input*/ from_input_offset_8163_to_tmp_pe_0, 
      /* input*/ from_input_offset_7683_to_tmp_pe_0, 
      /* input*/ from_input_offset_7203_to_tmp_pe_0, 
      /* input*/ from_input_offset_6723_to_tmp_pe_0, 
      /* input*/ from_input_offset_6243_to_tmp_pe_0, 
      /* input*/ from_input_offset_5763_to_tmp_pe_0, 
      /* input*/ from_input_offset_5283_to_tmp_pe_0, 
      /* input*/ from_input_offset_4803_to_tmp_pe_0, 
      /* input*/ from_input_offset_4323_to_tmp_pe_0, 
      /* input*/ from_input_offset_3843_to_tmp_pe_0, 
      /* input*/ from_input_offset_3363_to_tmp_pe_0, 
      /* input*/ from_input_offset_2883_to_tmp_pe_0, 
      /* input*/ from_input_offset_2403_to_tmp_pe_0, 
      /* input*/ from_input_offset_1923_to_tmp_pe_0, 
      /* input*/ from_input_offset_1443_to_tmp_pe_0, 
      /* input*/ from_input_offset_963_to_tmp_pe_0, 
      /* input*/ from_input_offset_483_to_tmp_pe_0, 
      /* input*/ from_input_offset_3_to_tmp_pe_0);
  Module11Func(
      /*output*/ from_tmp_offset_3_to_tmp_offset_7, 
      /*output*/ from_tmp_offset_3_to_output_pe_3, 
      /*output*/ from_tmp_offset_3_to_output_pe_2, 
      /*output*/ from_tmp_offset_3_to_output_pe_1, 
      /*output*/ from_tmp_offset_3_to_output_pe_0, 
      /* input*/ from_tmp_pe_0_to_tmp_offset_3);
  Module5Func(
      /*output*/ from_tmp_offset_7_to_tmp_offset_11, 
      /*output*/ from_tmp_offset_7_to_output_pe_3, 
      /*output*/ from_tmp_offset_7_to_output_pe_2, 
      /*output*/ from_tmp_offset_7_to_output_pe_1, 
      /*output*/ from_tmp_offset_7_to_output_pe_0, 
      /* input*/ from_tmp_offset_3_to_tmp_offset_7);
  Module5Func(
      /*output*/ from_tmp_offset_11_to_tmp_offset_15, 
      /*output*/ from_tmp_offset_11_to_output_pe_3, 
      /*output*/ from_tmp_offset_11_to_output_pe_2, 
      /*output*/ from_tmp_offset_11_to_output_pe_1, 
      /*output*/ from_tmp_offset_11_to_output_pe_0, 
      /* input*/ from_tmp_offset_7_to_tmp_offset_11);
  Module5Func(
      /*output*/ from_tmp_offset_15_to_tmp_offset_19, 
      /*output*/ from_tmp_offset_15_to_output_pe_3, 
      /*output*/ from_tmp_offset_15_to_output_pe_2, 
      /*output*/ from_tmp_offset_15_to_output_pe_1, 
      /*output*/ from_tmp_offset_15_to_output_pe_0, 
      /* input*/ from_tmp_offset_11_to_tmp_offset_15);
  Module4Func(
      /*output*/ from_output_pe_3_to_output_bank_1, 
      /* input*/ from_tmp_offset_18_to_output_pe_3, 
      /* input*/ from_tmp_offset_17_to_output_pe_3, 
      /* input*/ from_tmp_offset_16_to_output_pe_3, 
      /* input*/ from_tmp_offset_15_to_output_pe_3, 
      /* input*/ from_tmp_offset_14_to_output_pe_3, 
      /* input*/ from_tmp_offset_13_to_output_pe_3, 
      /* input*/ from_tmp_offset_12_to_output_pe_3, 
      /* input*/ from_tmp_offset_11_to_output_pe_3, 
      /* input*/ from_tmp_offset_10_to_output_pe_3, 
      /* input*/ from_tmp_offset_9_to_output_pe_3, 
      /* input*/ from_tmp_offset_8_to_output_pe_3, 
      /* input*/ from_tmp_offset_7_to_output_pe_3, 
      /* input*/ from_tmp_offset_6_to_output_pe_3, 
      /* input*/ from_tmp_offset_5_to_output_pe_3, 
      /* input*/ from_tmp_offset_4_to_output_pe_3, 
      /* input*/ from_tmp_offset_3_to_output_pe_3, 
      /* input*/ from_tmp_offset_2_to_output_pe_3, 
      /* input*/ from_tmp_offset_1_to_output_pe_3, 
      /* input*/ from_tmp_offset_0_to_output_pe_3);
  Module12Func(
      /*output*/ from_tmp_offset_19_to_output_pe_2, 
      /*output*/ from_tmp_offset_19_to_output_pe_1, 
      /*output*/ from_tmp_offset_19_to_output_pe_0, 
      /* input*/ from_tmp_offset_15_to_tmp_offset_19);
  Module4Func(
      /*output*/ from_output_pe_2_to_output_bank_1, 
      /* input*/ from_tmp_offset_19_to_output_pe_2, 
      /* input*/ from_tmp_offset_18_to_output_pe_2, 
      /* input*/ from_tmp_offset_17_to_output_pe_2, 
      /* input*/ from_tmp_offset_16_to_output_pe_2, 
      /* input*/ from_tmp_offset_15_to_output_pe_2, 
      /* input*/ from_tmp_offset_14_to_output_pe_2, 
      /* input*/ from_tmp_offset_13_to_output_pe_2, 
      /* input*/ from_tmp_offset_12_to_output_pe_2, 
      /* input*/ from_tmp_offset_11_to_output_pe_2, 
      /* input*/ from_tmp_offset_10_to_output_pe_2, 
      /* input*/ from_tmp_offset_9_to_output_pe_2, 
      /* input*/ from_tmp_offset_8_to_output_pe_2, 
      /* input*/ from_tmp_offset_7_to_output_pe_2, 
      /* input*/ from_tmp_offset_6_to_output_pe_2, 
      /* input*/ from_tmp_offset_5_to_output_pe_2, 
      /* input*/ from_tmp_offset_4_to_output_pe_2, 
      /* input*/ from_tmp_offset_3_to_output_pe_2, 
      /* input*/ from_tmp_offset_2_to_output_pe_2, 
      /* input*/ from_tmp_offset_1_to_output_pe_2);
  Module4Func(
      /*output*/ from_output_pe_1_to_output_bank_1, 
      /* input*/ from_tmp_offset_20_to_output_pe_1, 
      /* input*/ from_tmp_offset_19_to_output_pe_1, 
      /* input*/ from_tmp_offset_18_to_output_pe_1, 
      /* input*/ from_tmp_offset_17_to_output_pe_1, 
      /* input*/ from_tmp_offset_16_to_output_pe_1, 
      /* input*/ from_tmp_offset_15_to_output_pe_1, 
      /* input*/ from_tmp_offset_14_to_output_pe_1, 
      /* input*/ from_tmp_offset_13_to_output_pe_1, 
      /* input*/ from_tmp_offset_12_to_output_pe_1, 
      /* input*/ from_tmp_offset_11_to_output_pe_1, 
      /* input*/ from_tmp_offset_10_to_output_pe_1, 
      /* input*/ from_tmp_offset_9_to_output_pe_1, 
      /* input*/ from_tmp_offset_8_to_output_pe_1, 
      /* input*/ from_tmp_offset_7_to_output_pe_1, 
      /* input*/ from_tmp_offset_6_to_output_pe_1, 
      /* input*/ from_tmp_offset_5_to_output_pe_1, 
      /* input*/ from_tmp_offset_4_to_output_pe_1, 
      /* input*/ from_tmp_offset_3_to_output_pe_1, 
      /* input*/ from_tmp_offset_2_to_output_pe_1);
  Module4Func(
      /*output*/ from_output_pe_0_to_output_bank_1, 
      /* input*/ from_tmp_offset_21_to_output_pe_0, 
      /* input*/ from_tmp_offset_20_to_output_pe_0, 
      /* input*/ from_tmp_offset_19_to_output_pe_0, 
      /* input*/ from_tmp_offset_18_to_output_pe_0, 
      /* input*/ from_tmp_offset_17_to_output_pe_0, 
      /* input*/ from_tmp_offset_16_to_output_pe_0, 
      /* input*/ from_tmp_offset_15_to_output_pe_0, 
      /* input*/ from_tmp_offset_14_to_output_pe_0, 
      /* input*/ from_tmp_offset_13_to_output_pe_0, 
      /* input*/ from_tmp_offset_12_to_output_pe_0, 
      /* input*/ from_tmp_offset_11_to_output_pe_0, 
      /* input*/ from_tmp_offset_10_to_output_pe_0, 
      /* input*/ from_tmp_offset_9_to_output_pe_0, 
      /* input*/ from_tmp_offset_8_to_output_pe_0, 
      /* input*/ from_tmp_offset_7_to_output_pe_0, 
      /* input*/ from_tmp_offset_6_to_output_pe_0, 
      /* input*/ from_tmp_offset_5_to_output_pe_0, 
      /* input*/ from_tmp_offset_4_to_output_pe_0, 
      /* input*/ from_tmp_offset_3_to_output_pe_0);
  Module13Func(
      /*output*/ bank_1_output_buf, 
      /* input*/ from_output_pe_3_to_output_bank_1, 
      /* input*/ from_output_pe_2_to_output_bank_1, 
      /* input*/ from_output_pe_1_to_output_bank_1, 
      /* input*/ from_output_pe_0_to_output_bank_1);
  BurstWrite(bank_1_output, bank_1_output_buf, coalesced_data_num);
}

}  // extern "C"
