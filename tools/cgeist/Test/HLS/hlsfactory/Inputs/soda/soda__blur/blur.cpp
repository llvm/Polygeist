// this file can be generated from the following SODA DSL
/*
kernel: blur
burst width: 256
iterate: 1
unroll factor: 16
input dram 0 uint16: input(2000, *)
local uint16: blur_x(0, 0) = (input(0, 0) + input(0, 1) + input(0, 2)) / 3
output dram 1 uint16: blur_y(0, 0) = (blur_x(0, 0) + blur_x(1, 0) + blur_x(2, 0)) / 3
border: ignore
cluster: none
*/

// stencil window size: (3, 3)
// stencil distace: 4002
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
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_3, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_4, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_5, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_6, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_7, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_8, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_9, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_10, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_11, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_12, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_13, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_14, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_15, 
  /* input*/ hls::stream<Data<ap_uint<256>>>& dram_input_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_input_bank_0_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
#pragma HLS aggregate bit variable = fifo_st_6
#pragma HLS aggregate bit variable = fifo_st_7
#pragma HLS aggregate bit variable = fifo_st_8
#pragma HLS aggregate bit variable = fifo_st_9
#pragma HLS aggregate bit variable = fifo_st_10
#pragma HLS aggregate bit variable = fifo_st_11
#pragma HLS aggregate bit variable = fifo_st_12
#pragma HLS aggregate bit variable = fifo_st_13
#pragma HLS aggregate bit variable = fifo_st_14
#pragma HLS aggregate bit variable = fifo_st_15
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_0:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!dram_input_bank_0_fifo.empty())
    {
      ap_uint<256> dram_input_bank_0_buf;
      const bool dram_input_bank_0_buf_enable = ReadData(dram_input_bank_0_buf, dram_input_bank_0_fifo);
      const bool enabled = dram_input_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(255, 240))), enabled);
      WriteData(fifo_st_1, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(239, 224))), enabled);
      WriteData(fifo_st_2, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(223, 208))), enabled);
      WriteData(fifo_st_3, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(207, 192))), enabled);
      WriteData(fifo_st_4, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(191, 176))), enabled);
      WriteData(fifo_st_5, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(175, 160))), enabled);
      WriteData(fifo_st_6, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(159, 144))), enabled);
      WriteData(fifo_st_7, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(143, 128))), enabled);
      WriteData(fifo_st_8, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(127, 112))), enabled);
      WriteData(fifo_st_9, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(111, 96))), enabled);
      WriteData(fifo_st_10, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(95, 80))), enabled);
      WriteData(fifo_st_11, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(79, 64))), enabled);
      WriteData(fifo_st_12, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(63, 48))), enabled);
      WriteData(fifo_st_13, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(47, 32))), enabled);
      WriteData(fifo_st_14, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(31, 16))), enabled);
      WriteData(fifo_st_15, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(15, 0))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
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
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, uint16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<7> ptr_delay_125 = 0;
  uint16_t fifo_ref_0_delayed_125_buf[125];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_125_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_2:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_125_buf distance = 125
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_125 = fifo_ref_0_delayed_125_buf[ptr_delay_125];;
      const uint16_t let_0 = fifo_ref_0_delayed_125;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      WriteData(fifo_st_1, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_125_buf[ptr_delay_125] = fifo_ref_0;
      ptr_delay_125 < 124 ? (++ptr_delay_125) : (ptr_delay_125 = 0);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<7> ptr_delay_125 = 0;
  uint16_t fifo_ref_0_delayed_125_buf[125];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_125_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_3:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_125_buf distance = 125
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_125 = fifo_ref_0_delayed_125_buf[ptr_delay_125];;
      const uint16_t let_0 = fifo_ref_0_delayed_125;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_125_buf[ptr_delay_125] = fifo_ref_0;
      ptr_delay_125 < 124 ? (++ptr_delay_125) : (ptr_delay_125 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_2)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty())
    {
      uint16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      uint16_t fifo_ref_2;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t((uint16_t)((fifo_ref_0 + fifo_ref_1 + fifo_ref_2) / 3)), enabled);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_5:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_2, uint16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  uint16_t fifo_ref_0_delayed_1_buf[1];
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
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const uint16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      WriteData(fifo_st_1, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  uint16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_7:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const uint16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<ap_uint<256>>>& dram_blur_y_bank_1_fifo, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_6, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_7, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_8, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_9, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_10, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_11, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_12, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_13, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_14, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_15)
{
#pragma HLS aggregate bit variable = dram_blur_y_bank_1_fifo
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
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty() && !fifo_ld_7.empty() && !fifo_ld_8.empty() && !fifo_ld_9.empty() && !fifo_ld_10.empty() && !fifo_ld_11.empty() && !fifo_ld_12.empty() && !fifo_ld_13.empty() && !fifo_ld_14.empty() && !fifo_ld_15.empty())
    {
      uint16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      uint16_t fifo_ref_2;
      uint16_t fifo_ref_3;
      uint16_t fifo_ref_4;
      uint16_t fifo_ref_5;
      uint16_t fifo_ref_6;
      uint16_t fifo_ref_7;
      uint16_t fifo_ref_8;
      uint16_t fifo_ref_9;
      uint16_t fifo_ref_10;
      uint16_t fifo_ref_11;
      uint16_t fifo_ref_12;
      uint16_t fifo_ref_13;
      uint16_t fifo_ref_14;
      uint16_t fifo_ref_15;
      ap_uint<256> dram_blur_y_bank_1_buf;
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
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable && fifo_ref_7_enable && fifo_ref_8_enable && fifo_ref_9_enable && fifo_ref_10_enable && fifo_ref_11_enable && fifo_ref_12_enable && fifo_ref_13_enable && fifo_ref_14_enable && fifo_ref_15_enable;
      enable = enabled;
      dram_blur_y_bank_1_buf(255, 240) = Reinterpret<ap_uint<16>>(fifo_ref_0);
      dram_blur_y_bank_1_buf(239, 224) = Reinterpret<ap_uint<16>>(fifo_ref_1);
      dram_blur_y_bank_1_buf(223, 208) = Reinterpret<ap_uint<16>>(fifo_ref_2);
      dram_blur_y_bank_1_buf(207, 192) = Reinterpret<ap_uint<16>>(fifo_ref_3);
      dram_blur_y_bank_1_buf(191, 176) = Reinterpret<ap_uint<16>>(fifo_ref_4);
      dram_blur_y_bank_1_buf(175, 160) = Reinterpret<ap_uint<16>>(fifo_ref_5);
      dram_blur_y_bank_1_buf(159, 144) = Reinterpret<ap_uint<16>>(fifo_ref_6);
      dram_blur_y_bank_1_buf(143, 128) = Reinterpret<ap_uint<16>>(fifo_ref_7);
      dram_blur_y_bank_1_buf(127, 112) = Reinterpret<ap_uint<16>>(fifo_ref_8);
      dram_blur_y_bank_1_buf(111, 96) = Reinterpret<ap_uint<16>>(fifo_ref_9);
      dram_blur_y_bank_1_buf(95, 80) = Reinterpret<ap_uint<16>>(fifo_ref_10);
      dram_blur_y_bank_1_buf(79, 64) = Reinterpret<ap_uint<16>>(fifo_ref_11);
      dram_blur_y_bank_1_buf(63, 48) = Reinterpret<ap_uint<16>>(fifo_ref_12);
      dram_blur_y_bank_1_buf(47, 32) = Reinterpret<ap_uint<16>>(fifo_ref_13);
      dram_blur_y_bank_1_buf(31, 16) = Reinterpret<ap_uint<16>>(fifo_ref_14);
      dram_blur_y_bank_1_buf(15, 0) = Reinterpret<ap_uint<16>>(fifo_ref_15);
      WriteData(dram_blur_y_bank_1_fifo, dram_blur_y_bank_1_buf, enabled);
    } // if not empty
  } // for module_8
} // Module8Func

extern "C" {

void blur_kernel(
  ap_uint<256>* bank_1_blur_y, 
  ap_uint<256>* bank_0_input, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_blur_y offset = slave bundle = blur_y_bank_1
#pragma HLS interface m_axi port = bank_0_input offset = slave bundle = input_bank_0
#pragma HLS interface s_axilite port = bank_1_blur_y bundle = control
#pragma HLS interface s_axilite port = bank_0_input bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<256>>> bank_0_input_buf("bank_0_input_buf");
#pragma HLS stream variable = bank_0_input_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_input_buf
  hls::stream<Data<ap_uint<256>>> bank_1_blur_y_buf("bank_1_blur_y_buf");
#pragma HLS stream variable = bank_1_blur_y_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_blur_y_buf

  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_0("from_input_bank_0_to_input_offset_0");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_0
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_1("from_input_bank_0_to_input_offset_1");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_1
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_2("from_input_bank_0_to_input_offset_2");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_2
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_3("from_input_bank_0_to_input_offset_3");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_3
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_4("from_input_bank_0_to_input_offset_4");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_4 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_4
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_5("from_input_bank_0_to_input_offset_5");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_5 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_5
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_6("from_input_bank_0_to_input_offset_6");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_6 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_6
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_7("from_input_bank_0_to_input_offset_7");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_7 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_7
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_8("from_input_bank_0_to_input_offset_8");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_8 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_8
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_9("from_input_bank_0_to_input_offset_9");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_9 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_9
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_10("from_input_bank_0_to_input_offset_10");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_10 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_10
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_11("from_input_bank_0_to_input_offset_11");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_11 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_11
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_12("from_input_bank_0_to_input_offset_12");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_12 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_12
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_13("from_input_bank_0_to_input_offset_13");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_13 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_13
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_14("from_input_bank_0_to_input_offset_14");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_14 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_14
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_15("from_input_bank_0_to_input_offset_15");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_15 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_15
  hls::stream<Data<uint16_t>> from_input_offset_0_to_input_offset_2000("from_input_offset_0_to_input_offset_2000");
#pragma HLS stream variable = from_input_offset_0_to_input_offset_2000 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_0_to_input_offset_2000
  hls::stream<Data<uint16_t>> from_input_offset_0_to_blur_x_pe_15("from_input_offset_0_to_blur_x_pe_15");
#pragma HLS stream variable = from_input_offset_0_to_blur_x_pe_15 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_0_to_blur_x_pe_15
  hls::stream<Data<uint16_t>> from_input_offset_1_to_input_offset_2001("from_input_offset_1_to_input_offset_2001");
#pragma HLS stream variable = from_input_offset_1_to_input_offset_2001 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1_to_input_offset_2001
  hls::stream<Data<uint16_t>> from_input_offset_1_to_blur_x_pe_14("from_input_offset_1_to_blur_x_pe_14");
#pragma HLS stream variable = from_input_offset_1_to_blur_x_pe_14 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_1_to_blur_x_pe_14
  hls::stream<Data<uint16_t>> from_input_offset_2_to_input_offset_2002("from_input_offset_2_to_input_offset_2002");
#pragma HLS stream variable = from_input_offset_2_to_input_offset_2002 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2_to_input_offset_2002
  hls::stream<Data<uint16_t>> from_input_offset_2_to_blur_x_pe_13("from_input_offset_2_to_blur_x_pe_13");
#pragma HLS stream variable = from_input_offset_2_to_blur_x_pe_13 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_2_to_blur_x_pe_13
  hls::stream<Data<uint16_t>> from_input_offset_3_to_input_offset_2003("from_input_offset_3_to_input_offset_2003");
#pragma HLS stream variable = from_input_offset_3_to_input_offset_2003 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3_to_input_offset_2003
  hls::stream<Data<uint16_t>> from_input_offset_3_to_blur_x_pe_12("from_input_offset_3_to_blur_x_pe_12");
#pragma HLS stream variable = from_input_offset_3_to_blur_x_pe_12 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_3_to_blur_x_pe_12
  hls::stream<Data<uint16_t>> from_input_offset_4_to_input_offset_2004("from_input_offset_4_to_input_offset_2004");
#pragma HLS stream variable = from_input_offset_4_to_input_offset_2004 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4_to_input_offset_2004
  hls::stream<Data<uint16_t>> from_input_offset_4_to_blur_x_pe_11("from_input_offset_4_to_blur_x_pe_11");
#pragma HLS stream variable = from_input_offset_4_to_blur_x_pe_11 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_4_to_blur_x_pe_11
  hls::stream<Data<uint16_t>> from_input_offset_5_to_input_offset_2005("from_input_offset_5_to_input_offset_2005");
#pragma HLS stream variable = from_input_offset_5_to_input_offset_2005 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_5_to_input_offset_2005
  hls::stream<Data<uint16_t>> from_input_offset_5_to_blur_x_pe_10("from_input_offset_5_to_blur_x_pe_10");
#pragma HLS stream variable = from_input_offset_5_to_blur_x_pe_10 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_5_to_blur_x_pe_10
  hls::stream<Data<uint16_t>> from_input_offset_6_to_input_offset_2006("from_input_offset_6_to_input_offset_2006");
#pragma HLS stream variable = from_input_offset_6_to_input_offset_2006 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_6_to_input_offset_2006
  hls::stream<Data<uint16_t>> from_input_offset_6_to_blur_x_pe_9("from_input_offset_6_to_blur_x_pe_9");
#pragma HLS stream variable = from_input_offset_6_to_blur_x_pe_9 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_6_to_blur_x_pe_9
  hls::stream<Data<uint16_t>> from_input_offset_7_to_input_offset_2007("from_input_offset_7_to_input_offset_2007");
#pragma HLS stream variable = from_input_offset_7_to_input_offset_2007 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_7_to_input_offset_2007
  hls::stream<Data<uint16_t>> from_input_offset_7_to_blur_x_pe_8("from_input_offset_7_to_blur_x_pe_8");
#pragma HLS stream variable = from_input_offset_7_to_blur_x_pe_8 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_7_to_blur_x_pe_8
  hls::stream<Data<uint16_t>> from_input_offset_8_to_input_offset_2008("from_input_offset_8_to_input_offset_2008");
#pragma HLS stream variable = from_input_offset_8_to_input_offset_2008 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_8_to_input_offset_2008
  hls::stream<Data<uint16_t>> from_input_offset_8_to_blur_x_pe_7("from_input_offset_8_to_blur_x_pe_7");
#pragma HLS stream variable = from_input_offset_8_to_blur_x_pe_7 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_8_to_blur_x_pe_7
  hls::stream<Data<uint16_t>> from_input_offset_9_to_input_offset_2009("from_input_offset_9_to_input_offset_2009");
#pragma HLS stream variable = from_input_offset_9_to_input_offset_2009 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_9_to_input_offset_2009
  hls::stream<Data<uint16_t>> from_input_offset_9_to_blur_x_pe_6("from_input_offset_9_to_blur_x_pe_6");
#pragma HLS stream variable = from_input_offset_9_to_blur_x_pe_6 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_9_to_blur_x_pe_6
  hls::stream<Data<uint16_t>> from_input_offset_10_to_input_offset_2010("from_input_offset_10_to_input_offset_2010");
#pragma HLS stream variable = from_input_offset_10_to_input_offset_2010 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_10_to_input_offset_2010
  hls::stream<Data<uint16_t>> from_input_offset_10_to_blur_x_pe_5("from_input_offset_10_to_blur_x_pe_5");
#pragma HLS stream variable = from_input_offset_10_to_blur_x_pe_5 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_10_to_blur_x_pe_5
  hls::stream<Data<uint16_t>> from_input_offset_11_to_input_offset_2011("from_input_offset_11_to_input_offset_2011");
#pragma HLS stream variable = from_input_offset_11_to_input_offset_2011 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_11_to_input_offset_2011
  hls::stream<Data<uint16_t>> from_input_offset_11_to_blur_x_pe_4("from_input_offset_11_to_blur_x_pe_4");
#pragma HLS stream variable = from_input_offset_11_to_blur_x_pe_4 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_11_to_blur_x_pe_4
  hls::stream<Data<uint16_t>> from_input_offset_12_to_input_offset_2012("from_input_offset_12_to_input_offset_2012");
#pragma HLS stream variable = from_input_offset_12_to_input_offset_2012 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_12_to_input_offset_2012
  hls::stream<Data<uint16_t>> from_input_offset_12_to_blur_x_pe_3("from_input_offset_12_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_12_to_blur_x_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_12_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_13_to_input_offset_2013("from_input_offset_13_to_input_offset_2013");
#pragma HLS stream variable = from_input_offset_13_to_input_offset_2013 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_13_to_input_offset_2013
  hls::stream<Data<uint16_t>> from_input_offset_13_to_blur_x_pe_2("from_input_offset_13_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_13_to_blur_x_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_13_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_14_to_input_offset_2014("from_input_offset_14_to_input_offset_2014");
#pragma HLS stream variable = from_input_offset_14_to_input_offset_2014 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_14_to_input_offset_2014
  hls::stream<Data<uint16_t>> from_input_offset_14_to_blur_x_pe_1("from_input_offset_14_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_14_to_blur_x_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_14_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_15_to_input_offset_2015("from_input_offset_15_to_input_offset_2015");
#pragma HLS stream variable = from_input_offset_15_to_input_offset_2015 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_15_to_input_offset_2015
  hls::stream<Data<uint16_t>> from_input_offset_15_to_blur_x_pe_0("from_input_offset_15_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_15_to_blur_x_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_15_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_input_offset_2000_to_input_offset_4000("from_input_offset_2000_to_input_offset_4000");
#pragma HLS stream variable = from_input_offset_2000_to_input_offset_4000 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2000_to_input_offset_4000
  hls::stream<Data<uint16_t>> from_input_offset_2000_to_blur_x_pe_15("from_input_offset_2000_to_blur_x_pe_15");
#pragma HLS stream variable = from_input_offset_2000_to_blur_x_pe_15 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2000_to_blur_x_pe_15
  hls::stream<Data<uint16_t>> from_input_offset_2001_to_input_offset_4001("from_input_offset_2001_to_input_offset_4001");
#pragma HLS stream variable = from_input_offset_2001_to_input_offset_4001 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2001_to_input_offset_4001
  hls::stream<Data<uint16_t>> from_input_offset_2001_to_blur_x_pe_14("from_input_offset_2001_to_blur_x_pe_14");
#pragma HLS stream variable = from_input_offset_2001_to_blur_x_pe_14 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2001_to_blur_x_pe_14
  hls::stream<Data<uint16_t>> from_input_offset_2002_to_input_offset_4002("from_input_offset_2002_to_input_offset_4002");
#pragma HLS stream variable = from_input_offset_2002_to_input_offset_4002 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2002_to_input_offset_4002
  hls::stream<Data<uint16_t>> from_input_offset_2002_to_blur_x_pe_13("from_input_offset_2002_to_blur_x_pe_13");
#pragma HLS stream variable = from_input_offset_2002_to_blur_x_pe_13 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2002_to_blur_x_pe_13
  hls::stream<Data<uint16_t>> from_input_offset_2003_to_input_offset_4003("from_input_offset_2003_to_input_offset_4003");
#pragma HLS stream variable = from_input_offset_2003_to_input_offset_4003 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2003_to_input_offset_4003
  hls::stream<Data<uint16_t>> from_input_offset_2003_to_blur_x_pe_12("from_input_offset_2003_to_blur_x_pe_12");
#pragma HLS stream variable = from_input_offset_2003_to_blur_x_pe_12 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2003_to_blur_x_pe_12
  hls::stream<Data<uint16_t>> from_input_offset_2004_to_input_offset_4004("from_input_offset_2004_to_input_offset_4004");
#pragma HLS stream variable = from_input_offset_2004_to_input_offset_4004 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2004_to_input_offset_4004
  hls::stream<Data<uint16_t>> from_input_offset_2004_to_blur_x_pe_11("from_input_offset_2004_to_blur_x_pe_11");
#pragma HLS stream variable = from_input_offset_2004_to_blur_x_pe_11 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2004_to_blur_x_pe_11
  hls::stream<Data<uint16_t>> from_input_offset_2005_to_input_offset_4005("from_input_offset_2005_to_input_offset_4005");
#pragma HLS stream variable = from_input_offset_2005_to_input_offset_4005 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2005_to_input_offset_4005
  hls::stream<Data<uint16_t>> from_input_offset_2005_to_blur_x_pe_10("from_input_offset_2005_to_blur_x_pe_10");
#pragma HLS stream variable = from_input_offset_2005_to_blur_x_pe_10 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2005_to_blur_x_pe_10
  hls::stream<Data<uint16_t>> from_input_offset_2006_to_input_offset_4006("from_input_offset_2006_to_input_offset_4006");
#pragma HLS stream variable = from_input_offset_2006_to_input_offset_4006 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2006_to_input_offset_4006
  hls::stream<Data<uint16_t>> from_input_offset_2006_to_blur_x_pe_9("from_input_offset_2006_to_blur_x_pe_9");
#pragma HLS stream variable = from_input_offset_2006_to_blur_x_pe_9 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2006_to_blur_x_pe_9
  hls::stream<Data<uint16_t>> from_input_offset_2007_to_input_offset_4007("from_input_offset_2007_to_input_offset_4007");
#pragma HLS stream variable = from_input_offset_2007_to_input_offset_4007 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2007_to_input_offset_4007
  hls::stream<Data<uint16_t>> from_input_offset_2007_to_blur_x_pe_8("from_input_offset_2007_to_blur_x_pe_8");
#pragma HLS stream variable = from_input_offset_2007_to_blur_x_pe_8 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2007_to_blur_x_pe_8
  hls::stream<Data<uint16_t>> from_input_offset_2008_to_input_offset_4008("from_input_offset_2008_to_input_offset_4008");
#pragma HLS stream variable = from_input_offset_2008_to_input_offset_4008 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2008_to_input_offset_4008
  hls::stream<Data<uint16_t>> from_input_offset_2008_to_blur_x_pe_7("from_input_offset_2008_to_blur_x_pe_7");
#pragma HLS stream variable = from_input_offset_2008_to_blur_x_pe_7 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2008_to_blur_x_pe_7
  hls::stream<Data<uint16_t>> from_input_offset_2009_to_input_offset_4009("from_input_offset_2009_to_input_offset_4009");
#pragma HLS stream variable = from_input_offset_2009_to_input_offset_4009 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2009_to_input_offset_4009
  hls::stream<Data<uint16_t>> from_input_offset_2009_to_blur_x_pe_6("from_input_offset_2009_to_blur_x_pe_6");
#pragma HLS stream variable = from_input_offset_2009_to_blur_x_pe_6 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2009_to_blur_x_pe_6
  hls::stream<Data<uint16_t>> from_input_offset_2010_to_input_offset_4010("from_input_offset_2010_to_input_offset_4010");
#pragma HLS stream variable = from_input_offset_2010_to_input_offset_4010 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2010_to_input_offset_4010
  hls::stream<Data<uint16_t>> from_input_offset_2010_to_blur_x_pe_5("from_input_offset_2010_to_blur_x_pe_5");
#pragma HLS stream variable = from_input_offset_2010_to_blur_x_pe_5 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2010_to_blur_x_pe_5
  hls::stream<Data<uint16_t>> from_input_offset_2011_to_input_offset_4011("from_input_offset_2011_to_input_offset_4011");
#pragma HLS stream variable = from_input_offset_2011_to_input_offset_4011 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2011_to_input_offset_4011
  hls::stream<Data<uint16_t>> from_input_offset_2011_to_blur_x_pe_4("from_input_offset_2011_to_blur_x_pe_4");
#pragma HLS stream variable = from_input_offset_2011_to_blur_x_pe_4 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2011_to_blur_x_pe_4
  hls::stream<Data<uint16_t>> from_input_offset_2012_to_input_offset_4012("from_input_offset_2012_to_input_offset_4012");
#pragma HLS stream variable = from_input_offset_2012_to_input_offset_4012 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2012_to_input_offset_4012
  hls::stream<Data<uint16_t>> from_input_offset_2012_to_blur_x_pe_3("from_input_offset_2012_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_2012_to_blur_x_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2012_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_2013_to_input_offset_4013("from_input_offset_2013_to_input_offset_4013");
#pragma HLS stream variable = from_input_offset_2013_to_input_offset_4013 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2013_to_input_offset_4013
  hls::stream<Data<uint16_t>> from_input_offset_2013_to_blur_x_pe_2("from_input_offset_2013_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_2013_to_blur_x_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2013_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_2014_to_input_offset_4014("from_input_offset_2014_to_input_offset_4014");
#pragma HLS stream variable = from_input_offset_2014_to_input_offset_4014 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2014_to_input_offset_4014
  hls::stream<Data<uint16_t>> from_input_offset_2014_to_blur_x_pe_1("from_input_offset_2014_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_2014_to_blur_x_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2014_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_2015_to_input_offset_4015("from_input_offset_2015_to_input_offset_4015");
#pragma HLS stream variable = from_input_offset_2015_to_input_offset_4015 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2015_to_input_offset_4015
  hls::stream<Data<uint16_t>> from_input_offset_2015_to_blur_x_pe_0("from_input_offset_2015_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_2015_to_blur_x_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_2015_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_input_offset_4000_to_blur_x_pe_15("from_input_offset_4000_to_blur_x_pe_15");
#pragma HLS stream variable = from_input_offset_4000_to_blur_x_pe_15 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4000_to_blur_x_pe_15
  hls::stream<Data<uint16_t>> from_blur_x_pe_15_to_blur_x_offset_0("from_blur_x_pe_15_to_blur_x_offset_0");
#pragma HLS stream variable = from_blur_x_pe_15_to_blur_x_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_15_to_blur_x_offset_0
  hls::stream<Data<uint16_t>> from_blur_x_offset_0_to_blur_x_offset_16("from_blur_x_offset_0_to_blur_x_offset_16");
#pragma HLS stream variable = from_blur_x_offset_0_to_blur_x_offset_16 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_0_to_blur_x_offset_16
  hls::stream<Data<uint16_t>> from_blur_x_offset_0_to_blur_y_pe_15("from_blur_x_offset_0_to_blur_y_pe_15");
#pragma HLS stream variable = from_blur_x_offset_0_to_blur_y_pe_15 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_0_to_blur_y_pe_15
  hls::stream<Data<uint16_t>> from_input_offset_4001_to_blur_x_pe_14("from_input_offset_4001_to_blur_x_pe_14");
#pragma HLS stream variable = from_input_offset_4001_to_blur_x_pe_14 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4001_to_blur_x_pe_14
  hls::stream<Data<uint16_t>> from_blur_x_pe_14_to_blur_x_offset_1("from_blur_x_pe_14_to_blur_x_offset_1");
#pragma HLS stream variable = from_blur_x_pe_14_to_blur_x_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_14_to_blur_x_offset_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_x_offset_17("from_blur_x_offset_1_to_blur_x_offset_17");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_x_offset_17 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_x_offset_17
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_y_pe_15("from_blur_x_offset_1_to_blur_y_pe_15");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_y_pe_15 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_y_pe_15
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_y_pe_14("from_blur_x_offset_1_to_blur_y_pe_14");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_y_pe_14 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_y_pe_14
  hls::stream<Data<uint16_t>> from_input_offset_4002_to_blur_x_pe_13("from_input_offset_4002_to_blur_x_pe_13");
#pragma HLS stream variable = from_input_offset_4002_to_blur_x_pe_13 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4002_to_blur_x_pe_13
  hls::stream<Data<uint16_t>> from_blur_x_pe_13_to_blur_x_offset_2("from_blur_x_pe_13_to_blur_x_offset_2");
#pragma HLS stream variable = from_blur_x_pe_13_to_blur_x_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_13_to_blur_x_offset_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_15("from_blur_x_offset_2_to_blur_y_pe_15");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_15 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_15
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_14("from_blur_x_offset_2_to_blur_y_pe_14");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_14 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_14
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_13("from_blur_x_offset_2_to_blur_y_pe_13");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_13 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_13
  hls::stream<Data<uint16_t>> from_input_offset_4003_to_blur_x_pe_12("from_input_offset_4003_to_blur_x_pe_12");
#pragma HLS stream variable = from_input_offset_4003_to_blur_x_pe_12 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4003_to_blur_x_pe_12
  hls::stream<Data<uint16_t>> from_blur_x_pe_12_to_blur_x_offset_3("from_blur_x_pe_12_to_blur_x_offset_3");
#pragma HLS stream variable = from_blur_x_pe_12_to_blur_x_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_12_to_blur_x_offset_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_14("from_blur_x_offset_3_to_blur_y_pe_14");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_14 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_14
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_13("from_blur_x_offset_3_to_blur_y_pe_13");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_13 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_13
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_12("from_blur_x_offset_3_to_blur_y_pe_12");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_12 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_12
  hls::stream<Data<uint16_t>> from_input_offset_4004_to_blur_x_pe_11("from_input_offset_4004_to_blur_x_pe_11");
#pragma HLS stream variable = from_input_offset_4004_to_blur_x_pe_11 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4004_to_blur_x_pe_11
  hls::stream<Data<uint16_t>> from_blur_x_pe_11_to_blur_x_offset_4("from_blur_x_pe_11_to_blur_x_offset_4");
#pragma HLS stream variable = from_blur_x_pe_11_to_blur_x_offset_4 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_11_to_blur_x_offset_4
  hls::stream<Data<uint16_t>> from_blur_x_offset_4_to_blur_y_pe_13("from_blur_x_offset_4_to_blur_y_pe_13");
#pragma HLS stream variable = from_blur_x_offset_4_to_blur_y_pe_13 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_4_to_blur_y_pe_13
  hls::stream<Data<uint16_t>> from_blur_x_offset_4_to_blur_y_pe_12("from_blur_x_offset_4_to_blur_y_pe_12");
#pragma HLS stream variable = from_blur_x_offset_4_to_blur_y_pe_12 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_4_to_blur_y_pe_12
  hls::stream<Data<uint16_t>> from_blur_x_offset_4_to_blur_y_pe_11("from_blur_x_offset_4_to_blur_y_pe_11");
#pragma HLS stream variable = from_blur_x_offset_4_to_blur_y_pe_11 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_4_to_blur_y_pe_11
  hls::stream<Data<uint16_t>> from_input_offset_4005_to_blur_x_pe_10("from_input_offset_4005_to_blur_x_pe_10");
#pragma HLS stream variable = from_input_offset_4005_to_blur_x_pe_10 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4005_to_blur_x_pe_10
  hls::stream<Data<uint16_t>> from_blur_x_pe_10_to_blur_x_offset_5("from_blur_x_pe_10_to_blur_x_offset_5");
#pragma HLS stream variable = from_blur_x_pe_10_to_blur_x_offset_5 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_10_to_blur_x_offset_5
  hls::stream<Data<uint16_t>> from_blur_x_offset_5_to_blur_y_pe_12("from_blur_x_offset_5_to_blur_y_pe_12");
#pragma HLS stream variable = from_blur_x_offset_5_to_blur_y_pe_12 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_5_to_blur_y_pe_12
  hls::stream<Data<uint16_t>> from_blur_x_offset_5_to_blur_y_pe_11("from_blur_x_offset_5_to_blur_y_pe_11");
#pragma HLS stream variable = from_blur_x_offset_5_to_blur_y_pe_11 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_5_to_blur_y_pe_11
  hls::stream<Data<uint16_t>> from_blur_x_offset_5_to_blur_y_pe_10("from_blur_x_offset_5_to_blur_y_pe_10");
#pragma HLS stream variable = from_blur_x_offset_5_to_blur_y_pe_10 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_5_to_blur_y_pe_10
  hls::stream<Data<uint16_t>> from_input_offset_4006_to_blur_x_pe_9("from_input_offset_4006_to_blur_x_pe_9");
#pragma HLS stream variable = from_input_offset_4006_to_blur_x_pe_9 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4006_to_blur_x_pe_9
  hls::stream<Data<uint16_t>> from_blur_x_pe_9_to_blur_x_offset_6("from_blur_x_pe_9_to_blur_x_offset_6");
#pragma HLS stream variable = from_blur_x_pe_9_to_blur_x_offset_6 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_9_to_blur_x_offset_6
  hls::stream<Data<uint16_t>> from_blur_x_offset_6_to_blur_y_pe_11("from_blur_x_offset_6_to_blur_y_pe_11");
#pragma HLS stream variable = from_blur_x_offset_6_to_blur_y_pe_11 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_6_to_blur_y_pe_11
  hls::stream<Data<uint16_t>> from_blur_x_offset_6_to_blur_y_pe_10("from_blur_x_offset_6_to_blur_y_pe_10");
#pragma HLS stream variable = from_blur_x_offset_6_to_blur_y_pe_10 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_6_to_blur_y_pe_10
  hls::stream<Data<uint16_t>> from_blur_x_offset_6_to_blur_y_pe_9("from_blur_x_offset_6_to_blur_y_pe_9");
#pragma HLS stream variable = from_blur_x_offset_6_to_blur_y_pe_9 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_6_to_blur_y_pe_9
  hls::stream<Data<uint16_t>> from_input_offset_4007_to_blur_x_pe_8("from_input_offset_4007_to_blur_x_pe_8");
#pragma HLS stream variable = from_input_offset_4007_to_blur_x_pe_8 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4007_to_blur_x_pe_8
  hls::stream<Data<uint16_t>> from_blur_x_pe_8_to_blur_x_offset_7("from_blur_x_pe_8_to_blur_x_offset_7");
#pragma HLS stream variable = from_blur_x_pe_8_to_blur_x_offset_7 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_8_to_blur_x_offset_7
  hls::stream<Data<uint16_t>> from_blur_x_offset_7_to_blur_y_pe_10("from_blur_x_offset_7_to_blur_y_pe_10");
#pragma HLS stream variable = from_blur_x_offset_7_to_blur_y_pe_10 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_7_to_blur_y_pe_10
  hls::stream<Data<uint16_t>> from_blur_x_offset_7_to_blur_y_pe_9("from_blur_x_offset_7_to_blur_y_pe_9");
#pragma HLS stream variable = from_blur_x_offset_7_to_blur_y_pe_9 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_7_to_blur_y_pe_9
  hls::stream<Data<uint16_t>> from_blur_x_offset_7_to_blur_y_pe_8("from_blur_x_offset_7_to_blur_y_pe_8");
#pragma HLS stream variable = from_blur_x_offset_7_to_blur_y_pe_8 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_7_to_blur_y_pe_8
  hls::stream<Data<uint16_t>> from_input_offset_4008_to_blur_x_pe_7("from_input_offset_4008_to_blur_x_pe_7");
#pragma HLS stream variable = from_input_offset_4008_to_blur_x_pe_7 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4008_to_blur_x_pe_7
  hls::stream<Data<uint16_t>> from_blur_x_pe_7_to_blur_x_offset_8("from_blur_x_pe_7_to_blur_x_offset_8");
#pragma HLS stream variable = from_blur_x_pe_7_to_blur_x_offset_8 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_7_to_blur_x_offset_8
  hls::stream<Data<uint16_t>> from_blur_x_offset_8_to_blur_y_pe_9("from_blur_x_offset_8_to_blur_y_pe_9");
#pragma HLS stream variable = from_blur_x_offset_8_to_blur_y_pe_9 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_8_to_blur_y_pe_9
  hls::stream<Data<uint16_t>> from_blur_x_offset_8_to_blur_y_pe_8("from_blur_x_offset_8_to_blur_y_pe_8");
#pragma HLS stream variable = from_blur_x_offset_8_to_blur_y_pe_8 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_8_to_blur_y_pe_8
  hls::stream<Data<uint16_t>> from_blur_x_offset_8_to_blur_y_pe_7("from_blur_x_offset_8_to_blur_y_pe_7");
#pragma HLS stream variable = from_blur_x_offset_8_to_blur_y_pe_7 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_8_to_blur_y_pe_7
  hls::stream<Data<uint16_t>> from_input_offset_4009_to_blur_x_pe_6("from_input_offset_4009_to_blur_x_pe_6");
#pragma HLS stream variable = from_input_offset_4009_to_blur_x_pe_6 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4009_to_blur_x_pe_6
  hls::stream<Data<uint16_t>> from_blur_x_pe_6_to_blur_x_offset_9("from_blur_x_pe_6_to_blur_x_offset_9");
#pragma HLS stream variable = from_blur_x_pe_6_to_blur_x_offset_9 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_6_to_blur_x_offset_9
  hls::stream<Data<uint16_t>> from_blur_x_offset_9_to_blur_y_pe_8("from_blur_x_offset_9_to_blur_y_pe_8");
#pragma HLS stream variable = from_blur_x_offset_9_to_blur_y_pe_8 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_9_to_blur_y_pe_8
  hls::stream<Data<uint16_t>> from_blur_x_offset_9_to_blur_y_pe_7("from_blur_x_offset_9_to_blur_y_pe_7");
#pragma HLS stream variable = from_blur_x_offset_9_to_blur_y_pe_7 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_9_to_blur_y_pe_7
  hls::stream<Data<uint16_t>> from_blur_x_offset_9_to_blur_y_pe_6("from_blur_x_offset_9_to_blur_y_pe_6");
#pragma HLS stream variable = from_blur_x_offset_9_to_blur_y_pe_6 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_9_to_blur_y_pe_6
  hls::stream<Data<uint16_t>> from_input_offset_4010_to_blur_x_pe_5("from_input_offset_4010_to_blur_x_pe_5");
#pragma HLS stream variable = from_input_offset_4010_to_blur_x_pe_5 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4010_to_blur_x_pe_5
  hls::stream<Data<uint16_t>> from_blur_x_pe_5_to_blur_x_offset_10("from_blur_x_pe_5_to_blur_x_offset_10");
#pragma HLS stream variable = from_blur_x_pe_5_to_blur_x_offset_10 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_5_to_blur_x_offset_10
  hls::stream<Data<uint16_t>> from_blur_x_offset_10_to_blur_y_pe_7("from_blur_x_offset_10_to_blur_y_pe_7");
#pragma HLS stream variable = from_blur_x_offset_10_to_blur_y_pe_7 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_10_to_blur_y_pe_7
  hls::stream<Data<uint16_t>> from_blur_x_offset_10_to_blur_y_pe_6("from_blur_x_offset_10_to_blur_y_pe_6");
#pragma HLS stream variable = from_blur_x_offset_10_to_blur_y_pe_6 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_10_to_blur_y_pe_6
  hls::stream<Data<uint16_t>> from_blur_x_offset_10_to_blur_y_pe_5("from_blur_x_offset_10_to_blur_y_pe_5");
#pragma HLS stream variable = from_blur_x_offset_10_to_blur_y_pe_5 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_10_to_blur_y_pe_5
  hls::stream<Data<uint16_t>> from_input_offset_4011_to_blur_x_pe_4("from_input_offset_4011_to_blur_x_pe_4");
#pragma HLS stream variable = from_input_offset_4011_to_blur_x_pe_4 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4011_to_blur_x_pe_4
  hls::stream<Data<uint16_t>> from_blur_x_pe_4_to_blur_x_offset_11("from_blur_x_pe_4_to_blur_x_offset_11");
#pragma HLS stream variable = from_blur_x_pe_4_to_blur_x_offset_11 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_4_to_blur_x_offset_11
  hls::stream<Data<uint16_t>> from_blur_x_offset_11_to_blur_y_pe_6("from_blur_x_offset_11_to_blur_y_pe_6");
#pragma HLS stream variable = from_blur_x_offset_11_to_blur_y_pe_6 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_11_to_blur_y_pe_6
  hls::stream<Data<uint16_t>> from_blur_x_offset_11_to_blur_y_pe_5("from_blur_x_offset_11_to_blur_y_pe_5");
#pragma HLS stream variable = from_blur_x_offset_11_to_blur_y_pe_5 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_11_to_blur_y_pe_5
  hls::stream<Data<uint16_t>> from_blur_x_offset_11_to_blur_y_pe_4("from_blur_x_offset_11_to_blur_y_pe_4");
#pragma HLS stream variable = from_blur_x_offset_11_to_blur_y_pe_4 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_11_to_blur_y_pe_4
  hls::stream<Data<uint16_t>> from_input_offset_4012_to_blur_x_pe_3("from_input_offset_4012_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_4012_to_blur_x_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4012_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_pe_3_to_blur_x_offset_12("from_blur_x_pe_3_to_blur_x_offset_12");
#pragma HLS stream variable = from_blur_x_pe_3_to_blur_x_offset_12 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_3_to_blur_x_offset_12
  hls::stream<Data<uint16_t>> from_blur_x_offset_12_to_blur_y_pe_5("from_blur_x_offset_12_to_blur_y_pe_5");
#pragma HLS stream variable = from_blur_x_offset_12_to_blur_y_pe_5 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_12_to_blur_y_pe_5
  hls::stream<Data<uint16_t>> from_blur_x_offset_12_to_blur_y_pe_4("from_blur_x_offset_12_to_blur_y_pe_4");
#pragma HLS stream variable = from_blur_x_offset_12_to_blur_y_pe_4 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_12_to_blur_y_pe_4
  hls::stream<Data<uint16_t>> from_blur_x_offset_12_to_blur_y_pe_3("from_blur_x_offset_12_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_12_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_12_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_4013_to_blur_x_pe_2("from_input_offset_4013_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_4013_to_blur_x_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4013_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_pe_2_to_blur_x_offset_13("from_blur_x_pe_2_to_blur_x_offset_13");
#pragma HLS stream variable = from_blur_x_pe_2_to_blur_x_offset_13 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_2_to_blur_x_offset_13
  hls::stream<Data<uint16_t>> from_blur_x_offset_13_to_blur_y_pe_4("from_blur_x_offset_13_to_blur_y_pe_4");
#pragma HLS stream variable = from_blur_x_offset_13_to_blur_y_pe_4 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_13_to_blur_y_pe_4
  hls::stream<Data<uint16_t>> from_blur_x_offset_13_to_blur_y_pe_3("from_blur_x_offset_13_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_13_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_13_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_13_to_blur_y_pe_2("from_blur_x_offset_13_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_13_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_13_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_4014_to_blur_x_pe_1("from_input_offset_4014_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_4014_to_blur_x_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4014_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_pe_1_to_blur_x_offset_14("from_blur_x_pe_1_to_blur_x_offset_14");
#pragma HLS stream variable = from_blur_x_pe_1_to_blur_x_offset_14 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_1_to_blur_x_offset_14
  hls::stream<Data<uint16_t>> from_blur_x_offset_14_to_blur_y_pe_3("from_blur_x_offset_14_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_14_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_14_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_14_to_blur_y_pe_2("from_blur_x_offset_14_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_14_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_14_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_14_to_blur_y_pe_1("from_blur_x_offset_14_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_14_to_blur_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_14_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_4015_to_blur_x_pe_0("from_input_offset_4015_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_4015_to_blur_x_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_4015_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_blur_x_pe_0_to_blur_x_offset_15("from_blur_x_pe_0_to_blur_x_offset_15");
#pragma HLS stream variable = from_blur_x_pe_0_to_blur_x_offset_15 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_0_to_blur_x_offset_15
  hls::stream<Data<uint16_t>> from_blur_x_offset_15_to_blur_y_pe_2("from_blur_x_offset_15_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_15_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_15_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_15_to_blur_y_pe_1("from_blur_x_offset_15_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_15_to_blur_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_15_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_15_to_blur_y_pe_0("from_blur_x_offset_15_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_15_to_blur_y_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_15_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_x_offset_16_to_blur_y_pe_1("from_blur_x_offset_16_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_16_to_blur_y_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_16_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_16_to_blur_y_pe_0("from_blur_x_offset_16_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_16_to_blur_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_16_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_y_pe_15_to_blur_y_bank_1("from_blur_y_pe_15_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_15_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_15_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_17_to_blur_y_pe_0("from_blur_x_offset_17_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_17_to_blur_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_17_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_y_pe_14_to_blur_y_bank_1("from_blur_y_pe_14_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_14_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_14_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_13_to_blur_y_bank_1("from_blur_y_pe_13_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_13_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_13_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_12_to_blur_y_bank_1("from_blur_y_pe_12_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_12_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_12_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_11_to_blur_y_bank_1("from_blur_y_pe_11_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_11_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_11_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_10_to_blur_y_bank_1("from_blur_y_pe_10_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_10_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_10_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_9_to_blur_y_bank_1("from_blur_y_pe_9_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_9_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_9_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_8_to_blur_y_bank_1("from_blur_y_pe_8_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_8_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_8_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_7_to_blur_y_bank_1("from_blur_y_pe_7_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_7_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_7_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_6_to_blur_y_bank_1("from_blur_y_pe_6_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_6_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_6_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_5_to_blur_y_bank_1("from_blur_y_pe_5_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_5_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_5_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_4_to_blur_y_bank_1("from_blur_y_pe_4_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_4_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_4_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_3_to_blur_y_bank_1("from_blur_y_pe_3_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_3_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_3_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_2_to_blur_y_bank_1("from_blur_y_pe_2_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_2_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_2_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_1_to_blur_y_bank_1("from_blur_y_pe_1_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_1_to_blur_y_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_y_pe_1_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_0_to_blur_y_bank_1("from_blur_y_pe_0_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_0_to_blur_y_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_y_pe_0_to_blur_y_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_input_buf, bank_0_input, coalesced_data_num);
  Module0Func(
      /*output*/ from_input_bank_0_to_input_offset_0, 
      /*output*/ from_input_bank_0_to_input_offset_1, 
      /*output*/ from_input_bank_0_to_input_offset_2, 
      /*output*/ from_input_bank_0_to_input_offset_3, 
      /*output*/ from_input_bank_0_to_input_offset_4, 
      /*output*/ from_input_bank_0_to_input_offset_5, 
      /*output*/ from_input_bank_0_to_input_offset_6, 
      /*output*/ from_input_bank_0_to_input_offset_7, 
      /*output*/ from_input_bank_0_to_input_offset_8, 
      /*output*/ from_input_bank_0_to_input_offset_9, 
      /*output*/ from_input_bank_0_to_input_offset_10, 
      /*output*/ from_input_bank_0_to_input_offset_11, 
      /*output*/ from_input_bank_0_to_input_offset_12, 
      /*output*/ from_input_bank_0_to_input_offset_13, 
      /*output*/ from_input_bank_0_to_input_offset_14, 
      /*output*/ from_input_bank_0_to_input_offset_15, 
      /* input*/ bank_0_input_buf);
  Module1Func(
      /*output*/ from_input_offset_0_to_input_offset_2000, 
      /*output*/ from_input_offset_0_to_blur_x_pe_15, 
      /* input*/ from_input_bank_0_to_input_offset_0);
  Module1Func(
      /*output*/ from_input_offset_1_to_input_offset_2001, 
      /*output*/ from_input_offset_1_to_blur_x_pe_14, 
      /* input*/ from_input_bank_0_to_input_offset_1);
  Module1Func(
      /*output*/ from_input_offset_2_to_input_offset_2002, 
      /*output*/ from_input_offset_2_to_blur_x_pe_13, 
      /* input*/ from_input_bank_0_to_input_offset_2);
  Module1Func(
      /*output*/ from_input_offset_3_to_input_offset_2003, 
      /*output*/ from_input_offset_3_to_blur_x_pe_12, 
      /* input*/ from_input_bank_0_to_input_offset_3);
  Module1Func(
      /*output*/ from_input_offset_4_to_input_offset_2004, 
      /*output*/ from_input_offset_4_to_blur_x_pe_11, 
      /* input*/ from_input_bank_0_to_input_offset_4);
  Module1Func(
      /*output*/ from_input_offset_5_to_input_offset_2005, 
      /*output*/ from_input_offset_5_to_blur_x_pe_10, 
      /* input*/ from_input_bank_0_to_input_offset_5);
  Module1Func(
      /*output*/ from_input_offset_6_to_input_offset_2006, 
      /*output*/ from_input_offset_6_to_blur_x_pe_9, 
      /* input*/ from_input_bank_0_to_input_offset_6);
  Module1Func(
      /*output*/ from_input_offset_7_to_input_offset_2007, 
      /*output*/ from_input_offset_7_to_blur_x_pe_8, 
      /* input*/ from_input_bank_0_to_input_offset_7);
  Module1Func(
      /*output*/ from_input_offset_8_to_input_offset_2008, 
      /*output*/ from_input_offset_8_to_blur_x_pe_7, 
      /* input*/ from_input_bank_0_to_input_offset_8);
  Module1Func(
      /*output*/ from_input_offset_9_to_input_offset_2009, 
      /*output*/ from_input_offset_9_to_blur_x_pe_6, 
      /* input*/ from_input_bank_0_to_input_offset_9);
  Module1Func(
      /*output*/ from_input_offset_10_to_input_offset_2010, 
      /*output*/ from_input_offset_10_to_blur_x_pe_5, 
      /* input*/ from_input_bank_0_to_input_offset_10);
  Module1Func(
      /*output*/ from_input_offset_11_to_input_offset_2011, 
      /*output*/ from_input_offset_11_to_blur_x_pe_4, 
      /* input*/ from_input_bank_0_to_input_offset_11);
  Module1Func(
      /*output*/ from_input_offset_12_to_input_offset_2012, 
      /*output*/ from_input_offset_12_to_blur_x_pe_3, 
      /* input*/ from_input_bank_0_to_input_offset_12);
  Module1Func(
      /*output*/ from_input_offset_13_to_input_offset_2013, 
      /*output*/ from_input_offset_13_to_blur_x_pe_2, 
      /* input*/ from_input_bank_0_to_input_offset_13);
  Module1Func(
      /*output*/ from_input_offset_14_to_input_offset_2014, 
      /*output*/ from_input_offset_14_to_blur_x_pe_1, 
      /* input*/ from_input_bank_0_to_input_offset_14);
  Module1Func(
      /*output*/ from_input_offset_15_to_input_offset_2015, 
      /*output*/ from_input_offset_15_to_blur_x_pe_0, 
      /* input*/ from_input_bank_0_to_input_offset_15);
  Module2Func(
      /*output*/ from_input_offset_2000_to_input_offset_4000, 
      /*output*/ from_input_offset_2000_to_blur_x_pe_15, 
      /* input*/ from_input_offset_0_to_input_offset_2000);
  Module2Func(
      /*output*/ from_input_offset_2001_to_input_offset_4001, 
      /*output*/ from_input_offset_2001_to_blur_x_pe_14, 
      /* input*/ from_input_offset_1_to_input_offset_2001);
  Module2Func(
      /*output*/ from_input_offset_2002_to_input_offset_4002, 
      /*output*/ from_input_offset_2002_to_blur_x_pe_13, 
      /* input*/ from_input_offset_2_to_input_offset_2002);
  Module2Func(
      /*output*/ from_input_offset_2003_to_input_offset_4003, 
      /*output*/ from_input_offset_2003_to_blur_x_pe_12, 
      /* input*/ from_input_offset_3_to_input_offset_2003);
  Module2Func(
      /*output*/ from_input_offset_2004_to_input_offset_4004, 
      /*output*/ from_input_offset_2004_to_blur_x_pe_11, 
      /* input*/ from_input_offset_4_to_input_offset_2004);
  Module2Func(
      /*output*/ from_input_offset_2005_to_input_offset_4005, 
      /*output*/ from_input_offset_2005_to_blur_x_pe_10, 
      /* input*/ from_input_offset_5_to_input_offset_2005);
  Module2Func(
      /*output*/ from_input_offset_2006_to_input_offset_4006, 
      /*output*/ from_input_offset_2006_to_blur_x_pe_9, 
      /* input*/ from_input_offset_6_to_input_offset_2006);
  Module2Func(
      /*output*/ from_input_offset_2007_to_input_offset_4007, 
      /*output*/ from_input_offset_2007_to_blur_x_pe_8, 
      /* input*/ from_input_offset_7_to_input_offset_2007);
  Module2Func(
      /*output*/ from_input_offset_2008_to_input_offset_4008, 
      /*output*/ from_input_offset_2008_to_blur_x_pe_7, 
      /* input*/ from_input_offset_8_to_input_offset_2008);
  Module2Func(
      /*output*/ from_input_offset_2009_to_input_offset_4009, 
      /*output*/ from_input_offset_2009_to_blur_x_pe_6, 
      /* input*/ from_input_offset_9_to_input_offset_2009);
  Module2Func(
      /*output*/ from_input_offset_2010_to_input_offset_4010, 
      /*output*/ from_input_offset_2010_to_blur_x_pe_5, 
      /* input*/ from_input_offset_10_to_input_offset_2010);
  Module2Func(
      /*output*/ from_input_offset_2011_to_input_offset_4011, 
      /*output*/ from_input_offset_2011_to_blur_x_pe_4, 
      /* input*/ from_input_offset_11_to_input_offset_2011);
  Module2Func(
      /*output*/ from_input_offset_2012_to_input_offset_4012, 
      /*output*/ from_input_offset_2012_to_blur_x_pe_3, 
      /* input*/ from_input_offset_12_to_input_offset_2012);
  Module2Func(
      /*output*/ from_input_offset_2013_to_input_offset_4013, 
      /*output*/ from_input_offset_2013_to_blur_x_pe_2, 
      /* input*/ from_input_offset_13_to_input_offset_2013);
  Module2Func(
      /*output*/ from_input_offset_2014_to_input_offset_4014, 
      /*output*/ from_input_offset_2014_to_blur_x_pe_1, 
      /* input*/ from_input_offset_14_to_input_offset_2014);
  Module2Func(
      /*output*/ from_input_offset_2015_to_input_offset_4015, 
      /*output*/ from_input_offset_2015_to_blur_x_pe_0, 
      /* input*/ from_input_offset_15_to_input_offset_2015);
  Module3Func(
      /*output*/ from_input_offset_4000_to_blur_x_pe_15, 
      /* input*/ from_input_offset_2000_to_input_offset_4000);
  Module4Func(
      /*output*/ from_blur_x_pe_15_to_blur_x_offset_0, 
      /* input*/ from_input_offset_4000_to_blur_x_pe_15, 
      /* input*/ from_input_offset_2000_to_blur_x_pe_15, 
      /* input*/ from_input_offset_0_to_blur_x_pe_15);
  Module1Func(
      /*output*/ from_blur_x_offset_0_to_blur_x_offset_16, 
      /*output*/ from_blur_x_offset_0_to_blur_y_pe_15, 
      /* input*/ from_blur_x_pe_15_to_blur_x_offset_0);
  Module3Func(
      /*output*/ from_input_offset_4001_to_blur_x_pe_14, 
      /* input*/ from_input_offset_2001_to_input_offset_4001);
  Module4Func(
      /*output*/ from_blur_x_pe_14_to_blur_x_offset_1, 
      /* input*/ from_input_offset_4001_to_blur_x_pe_14, 
      /* input*/ from_input_offset_2001_to_blur_x_pe_14, 
      /* input*/ from_input_offset_1_to_blur_x_pe_14);
  Module5Func(
      /*output*/ from_blur_x_offset_1_to_blur_x_offset_17, 
      /*output*/ from_blur_x_offset_1_to_blur_y_pe_15, 
      /*output*/ from_blur_x_offset_1_to_blur_y_pe_14, 
      /* input*/ from_blur_x_pe_14_to_blur_x_offset_1);
  Module3Func(
      /*output*/ from_input_offset_4002_to_blur_x_pe_13, 
      /* input*/ from_input_offset_2002_to_input_offset_4002);
  Module4Func(
      /*output*/ from_blur_x_pe_13_to_blur_x_offset_2, 
      /* input*/ from_input_offset_4002_to_blur_x_pe_13, 
      /* input*/ from_input_offset_2002_to_blur_x_pe_13, 
      /* input*/ from_input_offset_2_to_blur_x_pe_13);
  Module5Func(
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_15, 
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_14, 
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_13, 
      /* input*/ from_blur_x_pe_13_to_blur_x_offset_2);
  Module3Func(
      /*output*/ from_input_offset_4003_to_blur_x_pe_12, 
      /* input*/ from_input_offset_2003_to_input_offset_4003);
  Module4Func(
      /*output*/ from_blur_x_pe_12_to_blur_x_offset_3, 
      /* input*/ from_input_offset_4003_to_blur_x_pe_12, 
      /* input*/ from_input_offset_2003_to_blur_x_pe_12, 
      /* input*/ from_input_offset_3_to_blur_x_pe_12);
  Module5Func(
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_14, 
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_13, 
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_12, 
      /* input*/ from_blur_x_pe_12_to_blur_x_offset_3);
  Module3Func(
      /*output*/ from_input_offset_4004_to_blur_x_pe_11, 
      /* input*/ from_input_offset_2004_to_input_offset_4004);
  Module4Func(
      /*output*/ from_blur_x_pe_11_to_blur_x_offset_4, 
      /* input*/ from_input_offset_4004_to_blur_x_pe_11, 
      /* input*/ from_input_offset_2004_to_blur_x_pe_11, 
      /* input*/ from_input_offset_4_to_blur_x_pe_11);
  Module5Func(
      /*output*/ from_blur_x_offset_4_to_blur_y_pe_13, 
      /*output*/ from_blur_x_offset_4_to_blur_y_pe_12, 
      /*output*/ from_blur_x_offset_4_to_blur_y_pe_11, 
      /* input*/ from_blur_x_pe_11_to_blur_x_offset_4);
  Module3Func(
      /*output*/ from_input_offset_4005_to_blur_x_pe_10, 
      /* input*/ from_input_offset_2005_to_input_offset_4005);
  Module4Func(
      /*output*/ from_blur_x_pe_10_to_blur_x_offset_5, 
      /* input*/ from_input_offset_4005_to_blur_x_pe_10, 
      /* input*/ from_input_offset_2005_to_blur_x_pe_10, 
      /* input*/ from_input_offset_5_to_blur_x_pe_10);
  Module5Func(
      /*output*/ from_blur_x_offset_5_to_blur_y_pe_12, 
      /*output*/ from_blur_x_offset_5_to_blur_y_pe_11, 
      /*output*/ from_blur_x_offset_5_to_blur_y_pe_10, 
      /* input*/ from_blur_x_pe_10_to_blur_x_offset_5);
  Module3Func(
      /*output*/ from_input_offset_4006_to_blur_x_pe_9, 
      /* input*/ from_input_offset_2006_to_input_offset_4006);
  Module4Func(
      /*output*/ from_blur_x_pe_9_to_blur_x_offset_6, 
      /* input*/ from_input_offset_4006_to_blur_x_pe_9, 
      /* input*/ from_input_offset_2006_to_blur_x_pe_9, 
      /* input*/ from_input_offset_6_to_blur_x_pe_9);
  Module5Func(
      /*output*/ from_blur_x_offset_6_to_blur_y_pe_11, 
      /*output*/ from_blur_x_offset_6_to_blur_y_pe_10, 
      /*output*/ from_blur_x_offset_6_to_blur_y_pe_9, 
      /* input*/ from_blur_x_pe_9_to_blur_x_offset_6);
  Module3Func(
      /*output*/ from_input_offset_4007_to_blur_x_pe_8, 
      /* input*/ from_input_offset_2007_to_input_offset_4007);
  Module4Func(
      /*output*/ from_blur_x_pe_8_to_blur_x_offset_7, 
      /* input*/ from_input_offset_4007_to_blur_x_pe_8, 
      /* input*/ from_input_offset_2007_to_blur_x_pe_8, 
      /* input*/ from_input_offset_7_to_blur_x_pe_8);
  Module5Func(
      /*output*/ from_blur_x_offset_7_to_blur_y_pe_10, 
      /*output*/ from_blur_x_offset_7_to_blur_y_pe_9, 
      /*output*/ from_blur_x_offset_7_to_blur_y_pe_8, 
      /* input*/ from_blur_x_pe_8_to_blur_x_offset_7);
  Module3Func(
      /*output*/ from_input_offset_4008_to_blur_x_pe_7, 
      /* input*/ from_input_offset_2008_to_input_offset_4008);
  Module4Func(
      /*output*/ from_blur_x_pe_7_to_blur_x_offset_8, 
      /* input*/ from_input_offset_4008_to_blur_x_pe_7, 
      /* input*/ from_input_offset_2008_to_blur_x_pe_7, 
      /* input*/ from_input_offset_8_to_blur_x_pe_7);
  Module5Func(
      /*output*/ from_blur_x_offset_8_to_blur_y_pe_9, 
      /*output*/ from_blur_x_offset_8_to_blur_y_pe_8, 
      /*output*/ from_blur_x_offset_8_to_blur_y_pe_7, 
      /* input*/ from_blur_x_pe_7_to_blur_x_offset_8);
  Module3Func(
      /*output*/ from_input_offset_4009_to_blur_x_pe_6, 
      /* input*/ from_input_offset_2009_to_input_offset_4009);
  Module4Func(
      /*output*/ from_blur_x_pe_6_to_blur_x_offset_9, 
      /* input*/ from_input_offset_4009_to_blur_x_pe_6, 
      /* input*/ from_input_offset_2009_to_blur_x_pe_6, 
      /* input*/ from_input_offset_9_to_blur_x_pe_6);
  Module5Func(
      /*output*/ from_blur_x_offset_9_to_blur_y_pe_8, 
      /*output*/ from_blur_x_offset_9_to_blur_y_pe_7, 
      /*output*/ from_blur_x_offset_9_to_blur_y_pe_6, 
      /* input*/ from_blur_x_pe_6_to_blur_x_offset_9);
  Module3Func(
      /*output*/ from_input_offset_4010_to_blur_x_pe_5, 
      /* input*/ from_input_offset_2010_to_input_offset_4010);
  Module4Func(
      /*output*/ from_blur_x_pe_5_to_blur_x_offset_10, 
      /* input*/ from_input_offset_4010_to_blur_x_pe_5, 
      /* input*/ from_input_offset_2010_to_blur_x_pe_5, 
      /* input*/ from_input_offset_10_to_blur_x_pe_5);
  Module5Func(
      /*output*/ from_blur_x_offset_10_to_blur_y_pe_7, 
      /*output*/ from_blur_x_offset_10_to_blur_y_pe_6, 
      /*output*/ from_blur_x_offset_10_to_blur_y_pe_5, 
      /* input*/ from_blur_x_pe_5_to_blur_x_offset_10);
  Module3Func(
      /*output*/ from_input_offset_4011_to_blur_x_pe_4, 
      /* input*/ from_input_offset_2011_to_input_offset_4011);
  Module4Func(
      /*output*/ from_blur_x_pe_4_to_blur_x_offset_11, 
      /* input*/ from_input_offset_4011_to_blur_x_pe_4, 
      /* input*/ from_input_offset_2011_to_blur_x_pe_4, 
      /* input*/ from_input_offset_11_to_blur_x_pe_4);
  Module5Func(
      /*output*/ from_blur_x_offset_11_to_blur_y_pe_6, 
      /*output*/ from_blur_x_offset_11_to_blur_y_pe_5, 
      /*output*/ from_blur_x_offset_11_to_blur_y_pe_4, 
      /* input*/ from_blur_x_pe_4_to_blur_x_offset_11);
  Module3Func(
      /*output*/ from_input_offset_4012_to_blur_x_pe_3, 
      /* input*/ from_input_offset_2012_to_input_offset_4012);
  Module4Func(
      /*output*/ from_blur_x_pe_3_to_blur_x_offset_12, 
      /* input*/ from_input_offset_4012_to_blur_x_pe_3, 
      /* input*/ from_input_offset_2012_to_blur_x_pe_3, 
      /* input*/ from_input_offset_12_to_blur_x_pe_3);
  Module5Func(
      /*output*/ from_blur_x_offset_12_to_blur_y_pe_5, 
      /*output*/ from_blur_x_offset_12_to_blur_y_pe_4, 
      /*output*/ from_blur_x_offset_12_to_blur_y_pe_3, 
      /* input*/ from_blur_x_pe_3_to_blur_x_offset_12);
  Module3Func(
      /*output*/ from_input_offset_4013_to_blur_x_pe_2, 
      /* input*/ from_input_offset_2013_to_input_offset_4013);
  Module4Func(
      /*output*/ from_blur_x_pe_2_to_blur_x_offset_13, 
      /* input*/ from_input_offset_4013_to_blur_x_pe_2, 
      /* input*/ from_input_offset_2013_to_blur_x_pe_2, 
      /* input*/ from_input_offset_13_to_blur_x_pe_2);
  Module5Func(
      /*output*/ from_blur_x_offset_13_to_blur_y_pe_4, 
      /*output*/ from_blur_x_offset_13_to_blur_y_pe_3, 
      /*output*/ from_blur_x_offset_13_to_blur_y_pe_2, 
      /* input*/ from_blur_x_pe_2_to_blur_x_offset_13);
  Module3Func(
      /*output*/ from_input_offset_4014_to_blur_x_pe_1, 
      /* input*/ from_input_offset_2014_to_input_offset_4014);
  Module4Func(
      /*output*/ from_blur_x_pe_1_to_blur_x_offset_14, 
      /* input*/ from_input_offset_4014_to_blur_x_pe_1, 
      /* input*/ from_input_offset_2014_to_blur_x_pe_1, 
      /* input*/ from_input_offset_14_to_blur_x_pe_1);
  Module5Func(
      /*output*/ from_blur_x_offset_14_to_blur_y_pe_3, 
      /*output*/ from_blur_x_offset_14_to_blur_y_pe_2, 
      /*output*/ from_blur_x_offset_14_to_blur_y_pe_1, 
      /* input*/ from_blur_x_pe_1_to_blur_x_offset_14);
  Module3Func(
      /*output*/ from_input_offset_4015_to_blur_x_pe_0, 
      /* input*/ from_input_offset_2015_to_input_offset_4015);
  Module4Func(
      /*output*/ from_blur_x_pe_0_to_blur_x_offset_15, 
      /* input*/ from_input_offset_4015_to_blur_x_pe_0, 
      /* input*/ from_input_offset_2015_to_blur_x_pe_0, 
      /* input*/ from_input_offset_15_to_blur_x_pe_0);
  Module5Func(
      /*output*/ from_blur_x_offset_15_to_blur_y_pe_2, 
      /*output*/ from_blur_x_offset_15_to_blur_y_pe_1, 
      /*output*/ from_blur_x_offset_15_to_blur_y_pe_0, 
      /* input*/ from_blur_x_pe_0_to_blur_x_offset_15);
  Module6Func(
      /*output*/ from_blur_x_offset_16_to_blur_y_pe_1, 
      /*output*/ from_blur_x_offset_16_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_0_to_blur_x_offset_16);
  Module4Func(
      /*output*/ from_blur_y_pe_15_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_15, 
      /* input*/ from_blur_x_offset_1_to_blur_y_pe_15, 
      /* input*/ from_blur_x_offset_0_to_blur_y_pe_15);
  Module7Func(
      /*output*/ from_blur_x_offset_17_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_1_to_blur_x_offset_17);
  Module4Func(
      /*output*/ from_blur_y_pe_14_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_14, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_14, 
      /* input*/ from_blur_x_offset_1_to_blur_y_pe_14);
  Module4Func(
      /*output*/ from_blur_y_pe_13_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_4_to_blur_y_pe_13, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_13, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_13);
  Module4Func(
      /*output*/ from_blur_y_pe_12_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_5_to_blur_y_pe_12, 
      /* input*/ from_blur_x_offset_4_to_blur_y_pe_12, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_12);
  Module4Func(
      /*output*/ from_blur_y_pe_11_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_6_to_blur_y_pe_11, 
      /* input*/ from_blur_x_offset_5_to_blur_y_pe_11, 
      /* input*/ from_blur_x_offset_4_to_blur_y_pe_11);
  Module4Func(
      /*output*/ from_blur_y_pe_10_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_7_to_blur_y_pe_10, 
      /* input*/ from_blur_x_offset_6_to_blur_y_pe_10, 
      /* input*/ from_blur_x_offset_5_to_blur_y_pe_10);
  Module4Func(
      /*output*/ from_blur_y_pe_9_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_8_to_blur_y_pe_9, 
      /* input*/ from_blur_x_offset_7_to_blur_y_pe_9, 
      /* input*/ from_blur_x_offset_6_to_blur_y_pe_9);
  Module4Func(
      /*output*/ from_blur_y_pe_8_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_9_to_blur_y_pe_8, 
      /* input*/ from_blur_x_offset_8_to_blur_y_pe_8, 
      /* input*/ from_blur_x_offset_7_to_blur_y_pe_8);
  Module4Func(
      /*output*/ from_blur_y_pe_7_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_10_to_blur_y_pe_7, 
      /* input*/ from_blur_x_offset_9_to_blur_y_pe_7, 
      /* input*/ from_blur_x_offset_8_to_blur_y_pe_7);
  Module4Func(
      /*output*/ from_blur_y_pe_6_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_11_to_blur_y_pe_6, 
      /* input*/ from_blur_x_offset_10_to_blur_y_pe_6, 
      /* input*/ from_blur_x_offset_9_to_blur_y_pe_6);
  Module4Func(
      /*output*/ from_blur_y_pe_5_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_12_to_blur_y_pe_5, 
      /* input*/ from_blur_x_offset_11_to_blur_y_pe_5, 
      /* input*/ from_blur_x_offset_10_to_blur_y_pe_5);
  Module4Func(
      /*output*/ from_blur_y_pe_4_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_13_to_blur_y_pe_4, 
      /* input*/ from_blur_x_offset_12_to_blur_y_pe_4, 
      /* input*/ from_blur_x_offset_11_to_blur_y_pe_4);
  Module4Func(
      /*output*/ from_blur_y_pe_3_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_14_to_blur_y_pe_3, 
      /* input*/ from_blur_x_offset_13_to_blur_y_pe_3, 
      /* input*/ from_blur_x_offset_12_to_blur_y_pe_3);
  Module4Func(
      /*output*/ from_blur_y_pe_2_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_15_to_blur_y_pe_2, 
      /* input*/ from_blur_x_offset_14_to_blur_y_pe_2, 
      /* input*/ from_blur_x_offset_13_to_blur_y_pe_2);
  Module4Func(
      /*output*/ from_blur_y_pe_1_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_16_to_blur_y_pe_1, 
      /* input*/ from_blur_x_offset_15_to_blur_y_pe_1, 
      /* input*/ from_blur_x_offset_14_to_blur_y_pe_1);
  Module4Func(
      /*output*/ from_blur_y_pe_0_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_17_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_16_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_15_to_blur_y_pe_0);
  Module8Func(
      /*output*/ bank_1_blur_y_buf, 
      /* input*/ from_blur_y_pe_15_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_14_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_13_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_12_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_11_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_10_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_9_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_8_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_7_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_6_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_5_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_4_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_3_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_2_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_1_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_0_to_blur_y_bank_1);
  BurstWrite(bank_1_blur_y, bank_1_blur_y_buf, coalesced_data_num);
}

}  // extern "C"
