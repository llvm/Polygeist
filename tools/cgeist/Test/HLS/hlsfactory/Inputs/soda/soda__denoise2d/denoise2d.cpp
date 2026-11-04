// this file can be generated from the following SODA DSL
/*
kernel: denoise2d
burst width: 128
iterate: 1
unroll factor: 4
input dram 0 float: f
input dram 1 float: u(32, *)
local float: diff_u(0, 0) = u(0, 0) - u(0, -1)
local float: diff_d(0, 0) = u(0, 0) - u(0, 1)
local float: diff_l(0, 0) = u(0, 0) - u(-1, 0)
local float: diff_r(0, 0) = u(0, 0) - u(1, 0)
local float: g(0, 0) = 1.0f / sqrt((1.0f + (diff_u(0, 0) * diff_u(0, 0)) + (diff_d(0, 0) * diff_d(0, 0)) + (diff_l(0, 0) * diff_l(0, 0)) + (diff_r(0, 0) * diff_r(0, 0))))
local float: r0(0, 0) = u(0, 0) * f(0, 0) * 4.9f
local float: r1(0, 0) = (r0(0, 0) * (2.5f + (r0(0, 0) * (10.2f + r0(0, 0))))) * (4.3f + (r0(0, 0) * (5.4f + (r0(0, 0) * (6.3f + r0(0, 0))))))
output dram 2 float: output(0, 0) = (u(0, 0) + (7.7f * ((u(0, 1) * g(0, 1)) + (u(0, -1) * g(0, -1)) + (u(-1, 0) * g(-1, 0)) + (u(1, 0) * g(1, 0)) + (5.7f * f(0, 0) * r1(0, 0))))) * (11.1f + (7.7f * (g(0, 1) + g(0, -1) + g(-1, 0) + g(1, 0) + 5.7f)))
border: ignore
cluster: none
*/

// stencil window size: (5, 5)
// stencil distace: 130
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
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<ap_uint<128>>>& dram_f_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_f_bank_0_fifo
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
    if (!dram_f_bank_0_fifo.empty())
    {
      ap_uint<128> dram_f_bank_0_buf;
      const bool dram_f_bank_0_buf_enable = ReadData(dram_f_bank_0_buf, dram_f_bank_0_fifo);
      const bool enabled = dram_f_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_f_bank_0_buf(127, 96))), enabled);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_f_bank_0_buf(95, 64))), enabled);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_f_bank_0_buf(63, 32))), enabled);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_f_bank_0_buf(31, 0))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<ap_uint<128>>>& dram_u_bank_1_fifo)
{
#pragma HLS aggregate bit variable = dram_u_bank_1_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_1:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!dram_u_bank_1_fifo.empty())
    {
      ap_uint<128> dram_u_bank_1_buf;
      const bool dram_u_bank_1_buf_enable = ReadData(dram_u_bank_1_buf, dram_u_bank_1_fifo);
      const bool enabled = dram_u_bank_1_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_u_bank_1_buf(127, 96))), enabled);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_u_bank_1_buf(95, 64))), enabled);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_u_bank_1_buf(63, 32))), enabled);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_u_bank_1_buf(31, 0))), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_2:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /*output*/ hls::stream<Data<float>>& fifo_st_5, 
  /*output*/ hls::stream<Data<float>>& fifo_st_6, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
#pragma HLS aggregate bit variable = fifo_st_6
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_3:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      WriteData(fifo_st_5, float(let_0), enabled);
      WriteData(fifo_st_6, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)(fifo_ref_0 - fifo_ref_1)), enabled);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /*output*/ hls::stream<Data<float>>& fifo_st_5, 
  /*output*/ hls::stream<Data<float>>& fifo_st_6, 
  /*output*/ hls::stream<Data<float>>& fifo_st_7, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
#pragma HLS aggregate bit variable = fifo_st_6
#pragma HLS aggregate bit variable = fifo_st_7
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_5:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      WriteData(fifo_st_5, float(let_0), enabled);
      WriteData(fifo_st_6, float(let_0), enabled);
      WriteData(fifo_st_7, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_7 = 0;
  float fifo_ref_0_delayed_7_buf[7];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_7_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_6:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_7_buf distance = 7
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const float let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
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
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /*output*/ hls::stream<Data<float>>& fifo_st_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_9:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      WriteData(fifo_st_5, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_10:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)(fifo_ref_0 * fifo_ref_1 * 4.9f)), enabled);
    } // if not empty
  } // for module_10
} // Module10Func

void Module11Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /*output*/ hls::stream<Data<float>>& fifo_st_5, 
  /*output*/ hls::stream<Data<float>>& fifo_st_6, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
#pragma HLS aggregate bit variable = fifo_st_6
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_11:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      WriteData(fifo_st_5, float(let_0), enabled);
      WriteData(fifo_st_6, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_11
} // Module11Func

void Module12Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_12:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)((fifo_ref_0 * (2.5f + (fifo_ref_0 * (10.2f + fifo_ref_0)))) * (4.3f + (fifo_ref_0 * (5.4f + (fifo_ref_0 * (6.3f + fifo_ref_0))))))), enabled);
    } // if not empty
  } // for module_12
} // Module12Func

void Module13Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_7 = 0;
  float fifo_ref_0_delayed_7_buf[7];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_7_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_13:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_7_buf distance = 7
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const float let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_13
} // Module13Func

void Module14Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_14:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)(1.0f / sqrt((1.0f + (fifo_ref_0 * fifo_ref_0) + (fifo_ref_1 * fifo_ref_1) + (fifo_ref_2 * fifo_ref_2) + (fifo_ref_3 * fifo_ref_3))))), enabled);
    } // if not empty
  } // for module_14
} // Module14Func

void Module15Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_15:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_15
} // Module15Func

void Module16Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_16:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      WriteData(fifo_st_4, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_16
} // Module16Func

void Module17Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_7 = 0;
  float fifo_ref_0_delayed_7_buf[7];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_7_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_17:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_7_buf distance = 7
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const float let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_17
} // Module17Func

void Module18Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_18:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_18
} // Module18Func

void Module19Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  float fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_19:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const float let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_19
} // Module19Func

void Module20Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_6, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_7, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_8, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_9, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_10)
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
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_20:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty() && !fifo_ld_7.empty() && !fifo_ld_8.empty() && !fifo_ld_9.empty() && !fifo_ld_10.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      float fifo_ref_5;
      float fifo_ref_6;
      float fifo_ref_7;
      float fifo_ref_8;
      float fifo_ref_9;
      float fifo_ref_10;
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
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable && fifo_ref_7_enable && fifo_ref_8_enable && fifo_ref_9_enable && fifo_ref_10_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)((fifo_ref_0 + (7.7f * ((fifo_ref_1 * fifo_ref_2) + (fifo_ref_3 * fifo_ref_4) + (fifo_ref_5 * fifo_ref_6) + (fifo_ref_7 * fifo_ref_8) + (5.7f * fifo_ref_9 * fifo_ref_10)))) * (11.1f + (7.7f * (fifo_ref_2 + fifo_ref_4 + fifo_ref_6 + fifo_ref_8 + 5.7f))))), enabled);
    } // if not empty
  } // for module_20
} // Module20Func

void Module21Func(
  /*output*/ hls::stream<Data<ap_uint<128>>>& dram_output_bank_2_fifo, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = dram_output_bank_2_fifo
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_21:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      ap_uint<128> dram_output_bank_2_buf;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      dram_output_bank_2_buf(95, 64) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_output_bank_2_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_output_bank_2_buf(127, 96) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_output_bank_2_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      WriteData(dram_output_bank_2_fifo, dram_output_bank_2_buf, enabled);
    } // if not empty
  } // for module_21
} // Module21Func

extern "C" {

void denoise2d_kernel(
  ap_uint<128>* bank_2_output, 
  ap_uint<128>* bank_0_f, 
  ap_uint<128>* bank_1_u, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_2_output offset = slave bundle = output_bank_2
#pragma HLS interface m_axi port = bank_0_f offset = slave bundle = f_bank_0
#pragma HLS interface m_axi port = bank_1_u offset = slave bundle = u_bank_1
#pragma HLS interface s_axilite port = bank_2_output bundle = control
#pragma HLS interface s_axilite port = bank_0_f bundle = control
#pragma HLS interface s_axilite port = bank_1_u bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<128>>> bank_0_f_buf("bank_0_f_buf");
#pragma HLS stream variable = bank_0_f_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_f_buf
  hls::stream<Data<ap_uint<128>>> bank_1_u_buf("bank_1_u_buf");
#pragma HLS stream variable = bank_1_u_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_u_buf
  hls::stream<Data<ap_uint<128>>> bank_2_output_buf("bank_2_output_buf");
#pragma HLS stream variable = bank_2_output_buf depth = 32
#pragma HLS aggregate bit variable = bank_2_output_buf

  hls::stream<Data<float>> from_f_bank_0_to_f_offset_0("from_f_bank_0_to_f_offset_0");
#pragma HLS stream variable = from_f_bank_0_to_f_offset_0 depth = 9
#pragma HLS aggregate bit variable = from_f_bank_0_to_f_offset_0
  hls::stream<Data<float>> from_f_bank_0_to_f_offset_1("from_f_bank_0_to_f_offset_1");
#pragma HLS stream variable = from_f_bank_0_to_f_offset_1 depth = 9
#pragma HLS aggregate bit variable = from_f_bank_0_to_f_offset_1
  hls::stream<Data<float>> from_f_bank_0_to_f_offset_2("from_f_bank_0_to_f_offset_2");
#pragma HLS stream variable = from_f_bank_0_to_f_offset_2 depth = 11
#pragma HLS aggregate bit variable = from_f_bank_0_to_f_offset_2
  hls::stream<Data<float>> from_f_bank_0_to_f_offset_3("from_f_bank_0_to_f_offset_3");
#pragma HLS stream variable = from_f_bank_0_to_f_offset_3 depth = 11
#pragma HLS aggregate bit variable = from_f_bank_0_to_f_offset_3
  hls::stream<Data<float>> from_u_bank_1_to_u_offset_0("from_u_bank_1_to_u_offset_0");
#pragma HLS stream variable = from_u_bank_1_to_u_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_u_bank_1_to_u_offset_0
  hls::stream<Data<float>> from_u_bank_1_to_u_offset_1("from_u_bank_1_to_u_offset_1");
#pragma HLS stream variable = from_u_bank_1_to_u_offset_1 depth = 5
#pragma HLS aggregate bit variable = from_u_bank_1_to_u_offset_1
  hls::stream<Data<float>> from_u_bank_1_to_u_offset_2("from_u_bank_1_to_u_offset_2");
#pragma HLS stream variable = from_u_bank_1_to_u_offset_2 depth = 7
#pragma HLS aggregate bit variable = from_u_bank_1_to_u_offset_2
  hls::stream<Data<float>> from_u_bank_1_to_u_offset_3("from_u_bank_1_to_u_offset_3");
#pragma HLS stream variable = from_u_bank_1_to_u_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_u_bank_1_to_u_offset_3
  hls::stream<Data<float>> from_f_offset_0_to_r0_pe_3("from_f_offset_0_to_r0_pe_3");
#pragma HLS stream variable = from_f_offset_0_to_r0_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_f_offset_0_to_r0_pe_3
  hls::stream<Data<float>> from_f_offset_0_to_output_pe_3("from_f_offset_0_to_output_pe_3");
#pragma HLS stream variable = from_f_offset_0_to_output_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_f_offset_0_to_output_pe_3
  hls::stream<Data<float>> from_f_offset_1_to_r0_pe_2("from_f_offset_1_to_r0_pe_2");
#pragma HLS stream variable = from_f_offset_1_to_r0_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_f_offset_1_to_r0_pe_2
  hls::stream<Data<float>> from_f_offset_1_to_output_pe_2("from_f_offset_1_to_output_pe_2");
#pragma HLS stream variable = from_f_offset_1_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_f_offset_1_to_output_pe_2
  hls::stream<Data<float>> from_f_offset_2_to_r0_pe_1("from_f_offset_2_to_r0_pe_1");
#pragma HLS stream variable = from_f_offset_2_to_r0_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_f_offset_2_to_r0_pe_1
  hls::stream<Data<float>> from_f_offset_2_to_output_pe_1("from_f_offset_2_to_output_pe_1");
#pragma HLS stream variable = from_f_offset_2_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_f_offset_2_to_output_pe_1
  hls::stream<Data<float>> from_f_offset_3_to_r0_pe_0("from_f_offset_3_to_r0_pe_0");
#pragma HLS stream variable = from_f_offset_3_to_r0_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_f_offset_3_to_r0_pe_0
  hls::stream<Data<float>> from_f_offset_3_to_output_pe_0("from_f_offset_3_to_output_pe_0");
#pragma HLS stream variable = from_f_offset_3_to_output_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_f_offset_3_to_output_pe_0
  hls::stream<Data<float>> from_u_offset_0_to_u_offset_32("from_u_offset_0_to_u_offset_32");
#pragma HLS stream variable = from_u_offset_0_to_u_offset_32 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_0_to_u_offset_32
  hls::stream<Data<float>> from_u_offset_0_to_diff_d_pe_3("from_u_offset_0_to_diff_d_pe_3");
#pragma HLS stream variable = from_u_offset_0_to_diff_d_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_0_to_diff_d_pe_3
  hls::stream<Data<float>> from_u_offset_1_to_u_offset_33("from_u_offset_1_to_u_offset_33");
#pragma HLS stream variable = from_u_offset_1_to_u_offset_33 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_1_to_u_offset_33
  hls::stream<Data<float>> from_u_offset_1_to_diff_d_pe_2("from_u_offset_1_to_diff_d_pe_2");
#pragma HLS stream variable = from_u_offset_1_to_diff_d_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_1_to_diff_d_pe_2
  hls::stream<Data<float>> from_u_offset_2_to_u_offset_34("from_u_offset_2_to_u_offset_34");
#pragma HLS stream variable = from_u_offset_2_to_u_offset_34 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_2_to_u_offset_34
  hls::stream<Data<float>> from_u_offset_2_to_diff_d_pe_1("from_u_offset_2_to_diff_d_pe_1");
#pragma HLS stream variable = from_u_offset_2_to_diff_d_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_2_to_diff_d_pe_1
  hls::stream<Data<float>> from_u_offset_3_to_u_offset_31("from_u_offset_3_to_u_offset_31");
#pragma HLS stream variable = from_u_offset_3_to_u_offset_31 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_3_to_u_offset_31
  hls::stream<Data<float>> from_u_offset_3_to_diff_d_pe_0("from_u_offset_3_to_diff_d_pe_0");
#pragma HLS stream variable = from_u_offset_3_to_diff_d_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_u_offset_3_to_diff_d_pe_0
  hls::stream<Data<float>> from_u_offset_32_to_u_offset_36("from_u_offset_32_to_u_offset_36");
#pragma HLS stream variable = from_u_offset_32_to_u_offset_36 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_32_to_u_offset_36
  hls::stream<Data<float>> from_u_offset_32_to_diff_d_pe_3("from_u_offset_32_to_diff_d_pe_3");
#pragma HLS stream variable = from_u_offset_32_to_diff_d_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_32_to_diff_d_pe_3
  hls::stream<Data<float>> from_u_offset_32_to_diff_l_pe_3("from_u_offset_32_to_diff_l_pe_3");
#pragma HLS stream variable = from_u_offset_32_to_diff_l_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_32_to_diff_l_pe_3
  hls::stream<Data<float>> from_u_offset_32_to_diff_r_pe_3("from_u_offset_32_to_diff_r_pe_3");
#pragma HLS stream variable = from_u_offset_32_to_diff_r_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_32_to_diff_r_pe_3
  hls::stream<Data<float>> from_u_offset_32_to_diff_r_pe_2("from_u_offset_32_to_diff_r_pe_2");
#pragma HLS stream variable = from_u_offset_32_to_diff_r_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_32_to_diff_r_pe_2
  hls::stream<Data<float>> from_u_offset_32_to_diff_u_pe_3("from_u_offset_32_to_diff_u_pe_3");
#pragma HLS stream variable = from_u_offset_32_to_diff_u_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_u_offset_32_to_diff_u_pe_3
  hls::stream<Data<float>> from_u_offset_32_to_output_pe_3("from_u_offset_32_to_output_pe_3");
#pragma HLS stream variable = from_u_offset_32_to_output_pe_3 depth = 17
#pragma HLS aggregate bit variable = from_u_offset_32_to_output_pe_3
  hls::stream<Data<float>> from_diff_d_pe_3_to_diff_d_offset_0("from_diff_d_pe_3_to_diff_d_offset_0");
#pragma HLS stream variable = from_diff_d_pe_3_to_diff_d_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_diff_d_pe_3_to_diff_d_offset_0
  hls::stream<Data<float>> from_u_offset_33_to_u_offset_65("from_u_offset_33_to_u_offset_65");
#pragma HLS stream variable = from_u_offset_33_to_u_offset_65 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_33_to_u_offset_65
  hls::stream<Data<float>> from_u_offset_33_to_diff_d_pe_2("from_u_offset_33_to_diff_d_pe_2");
#pragma HLS stream variable = from_u_offset_33_to_diff_d_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_d_pe_2
  hls::stream<Data<float>> from_u_offset_33_to_diff_l_pe_3("from_u_offset_33_to_diff_l_pe_3");
#pragma HLS stream variable = from_u_offset_33_to_diff_l_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_l_pe_3
  hls::stream<Data<float>> from_u_offset_33_to_diff_l_pe_2("from_u_offset_33_to_diff_l_pe_2");
#pragma HLS stream variable = from_u_offset_33_to_diff_l_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_l_pe_2
  hls::stream<Data<float>> from_u_offset_33_to_diff_r_pe_2("from_u_offset_33_to_diff_r_pe_2");
#pragma HLS stream variable = from_u_offset_33_to_diff_r_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_r_pe_2
  hls::stream<Data<float>> from_u_offset_33_to_diff_r_pe_1("from_u_offset_33_to_diff_r_pe_1");
#pragma HLS stream variable = from_u_offset_33_to_diff_r_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_r_pe_1
  hls::stream<Data<float>> from_u_offset_33_to_diff_u_pe_2("from_u_offset_33_to_diff_u_pe_2");
#pragma HLS stream variable = from_u_offset_33_to_diff_u_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_33_to_diff_u_pe_2
  hls::stream<Data<float>> from_u_offset_33_to_output_pe_2("from_u_offset_33_to_output_pe_2");
#pragma HLS stream variable = from_u_offset_33_to_output_pe_2 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_33_to_output_pe_2
  hls::stream<Data<float>> from_diff_d_pe_2_to_diff_d_offset_1("from_diff_d_pe_2_to_diff_d_offset_1");
#pragma HLS stream variable = from_diff_d_pe_2_to_diff_d_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_diff_d_pe_2_to_diff_d_offset_1
  hls::stream<Data<float>> from_u_offset_34_to_u_offset_66("from_u_offset_34_to_u_offset_66");
#pragma HLS stream variable = from_u_offset_34_to_u_offset_66 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_u_offset_66
  hls::stream<Data<float>> from_u_offset_34_to_diff_d_pe_1("from_u_offset_34_to_diff_d_pe_1");
#pragma HLS stream variable = from_u_offset_34_to_diff_d_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_d_pe_1
  hls::stream<Data<float>> from_u_offset_34_to_diff_l_pe_2("from_u_offset_34_to_diff_l_pe_2");
#pragma HLS stream variable = from_u_offset_34_to_diff_l_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_l_pe_2
  hls::stream<Data<float>> from_u_offset_34_to_diff_l_pe_1("from_u_offset_34_to_diff_l_pe_1");
#pragma HLS stream variable = from_u_offset_34_to_diff_l_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_l_pe_1
  hls::stream<Data<float>> from_u_offset_34_to_diff_r_pe_1("from_u_offset_34_to_diff_r_pe_1");
#pragma HLS stream variable = from_u_offset_34_to_diff_r_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_r_pe_1
  hls::stream<Data<float>> from_u_offset_34_to_diff_r_pe_0("from_u_offset_34_to_diff_r_pe_0");
#pragma HLS stream variable = from_u_offset_34_to_diff_r_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_r_pe_0
  hls::stream<Data<float>> from_u_offset_34_to_diff_u_pe_1("from_u_offset_34_to_diff_u_pe_1");
#pragma HLS stream variable = from_u_offset_34_to_diff_u_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_34_to_diff_u_pe_1
  hls::stream<Data<float>> from_u_offset_34_to_output_pe_1("from_u_offset_34_to_output_pe_1");
#pragma HLS stream variable = from_u_offset_34_to_output_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_34_to_output_pe_1
  hls::stream<Data<float>> from_diff_d_pe_1_to_diff_d_offset_2("from_diff_d_pe_1_to_diff_d_offset_2");
#pragma HLS stream variable = from_diff_d_pe_1_to_diff_d_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_d_pe_1_to_diff_d_offset_2
  hls::stream<Data<float>> from_u_offset_31_to_u_offset_35("from_u_offset_31_to_u_offset_35");
#pragma HLS stream variable = from_u_offset_31_to_u_offset_35 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_31_to_u_offset_35
  hls::stream<Data<float>> from_u_offset_31_to_diff_r_pe_3("from_u_offset_31_to_diff_r_pe_3");
#pragma HLS stream variable = from_u_offset_31_to_diff_r_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_31_to_diff_r_pe_3
  hls::stream<Data<float>> from_u_offset_36_to_u_offset_64("from_u_offset_36_to_u_offset_64");
#pragma HLS stream variable = from_u_offset_36_to_u_offset_64 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_36_to_u_offset_64
  hls::stream<Data<float>> from_u_offset_36_to_diff_l_pe_0("from_u_offset_36_to_diff_l_pe_0");
#pragma HLS stream variable = from_u_offset_36_to_diff_l_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_36_to_diff_l_pe_0
  hls::stream<Data<float>> from_diff_l_pe_3_to_diff_l_offset_0("from_diff_l_pe_3_to_diff_l_offset_0");
#pragma HLS stream variable = from_diff_l_pe_3_to_diff_l_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_diff_l_pe_3_to_diff_l_offset_0
  hls::stream<Data<float>> from_diff_r_pe_3_to_diff_r_offset_0("from_diff_r_pe_3_to_diff_r_offset_0");
#pragma HLS stream variable = from_diff_r_pe_3_to_diff_r_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_diff_r_pe_3_to_diff_r_offset_0
  hls::stream<Data<float>> from_diff_r_pe_2_to_diff_r_offset_1("from_diff_r_pe_2_to_diff_r_offset_1");
#pragma HLS stream variable = from_diff_r_pe_2_to_diff_r_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_diff_r_pe_2_to_diff_r_offset_1
  hls::stream<Data<float>> from_diff_d_offset_0_to_g_pe_3("from_diff_d_offset_0_to_g_pe_3");
#pragma HLS stream variable = from_diff_d_offset_0_to_g_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_diff_d_offset_0_to_g_pe_3
  hls::stream<Data<float>> from_u_offset_65_to_u_offset_97("from_u_offset_65_to_u_offset_97");
#pragma HLS stream variable = from_u_offset_65_to_u_offset_97 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_65_to_u_offset_97
  hls::stream<Data<float>> from_u_offset_65_to_diff_u_pe_2("from_u_offset_65_to_diff_u_pe_2");
#pragma HLS stream variable = from_u_offset_65_to_diff_u_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_65_to_diff_u_pe_2
  hls::stream<Data<float>> from_u_offset_65_to_r0_pe_2("from_u_offset_65_to_r0_pe_2");
#pragma HLS stream variable = from_u_offset_65_to_r0_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_65_to_r0_pe_2
  hls::stream<Data<float>> from_u_offset_65_to_output_pe_3("from_u_offset_65_to_output_pe_3");
#pragma HLS stream variable = from_u_offset_65_to_output_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_65_to_output_pe_3
  hls::stream<Data<float>> from_u_offset_65_to_output_pe_2("from_u_offset_65_to_output_pe_2");
#pragma HLS stream variable = from_u_offset_65_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_u_offset_65_to_output_pe_2
  hls::stream<Data<float>> from_u_offset_65_to_output_pe_1("from_u_offset_65_to_output_pe_1");
#pragma HLS stream variable = from_u_offset_65_to_output_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_65_to_output_pe_1
  hls::stream<Data<float>> from_r0_pe_2_to_r0_offset_1("from_r0_pe_2_to_r0_offset_1");
#pragma HLS stream variable = from_r0_pe_2_to_r0_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_r0_pe_2_to_r0_offset_1
  hls::stream<Data<float>> from_r0_offset_1_to_r1_pe_2("from_r0_offset_1_to_r1_pe_2");
#pragma HLS stream variable = from_r0_offset_1_to_r1_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_r0_offset_1_to_r1_pe_2
  hls::stream<Data<float>> from_diff_l_pe_2_to_diff_l_offset_1("from_diff_l_pe_2_to_diff_l_offset_1");
#pragma HLS stream variable = from_diff_l_pe_2_to_diff_l_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_diff_l_pe_2_to_diff_l_offset_1
  hls::stream<Data<float>> from_diff_r_pe_1_to_diff_r_offset_2("from_diff_r_pe_1_to_diff_r_offset_2");
#pragma HLS stream variable = from_diff_r_pe_1_to_diff_r_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_r_pe_1_to_diff_r_offset_2
  hls::stream<Data<float>> from_diff_u_pe_2_to_diff_u_offset_1("from_diff_u_pe_2_to_diff_u_offset_1");
#pragma HLS stream variable = from_diff_u_pe_2_to_diff_u_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_pe_2_to_diff_u_offset_1
  hls::stream<Data<float>> from_diff_d_offset_1_to_g_pe_2("from_diff_d_offset_1_to_g_pe_2");
#pragma HLS stream variable = from_diff_d_offset_1_to_g_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_diff_d_offset_1_to_g_pe_2
  hls::stream<Data<float>> from_u_offset_66_to_u_offset_98("from_u_offset_66_to_u_offset_98");
#pragma HLS stream variable = from_u_offset_66_to_u_offset_98 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_66_to_u_offset_98
  hls::stream<Data<float>> from_u_offset_66_to_diff_u_pe_1("from_u_offset_66_to_diff_u_pe_1");
#pragma HLS stream variable = from_u_offset_66_to_diff_u_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_66_to_diff_u_pe_1
  hls::stream<Data<float>> from_u_offset_66_to_r0_pe_1("from_u_offset_66_to_r0_pe_1");
#pragma HLS stream variable = from_u_offset_66_to_r0_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_66_to_r0_pe_1
  hls::stream<Data<float>> from_u_offset_66_to_output_pe_2("from_u_offset_66_to_output_pe_2");
#pragma HLS stream variable = from_u_offset_66_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_u_offset_66_to_output_pe_2
  hls::stream<Data<float>> from_u_offset_66_to_output_pe_1("from_u_offset_66_to_output_pe_1");
#pragma HLS stream variable = from_u_offset_66_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_u_offset_66_to_output_pe_1
  hls::stream<Data<float>> from_u_offset_66_to_output_pe_0("from_u_offset_66_to_output_pe_0");
#pragma HLS stream variable = from_u_offset_66_to_output_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_66_to_output_pe_0
  hls::stream<Data<float>> from_r0_pe_1_to_r0_offset_2("from_r0_pe_1_to_r0_offset_2");
#pragma HLS stream variable = from_r0_pe_1_to_r0_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_r0_pe_1_to_r0_offset_2
  hls::stream<Data<float>> from_r0_offset_2_to_r1_pe_1("from_r0_offset_2_to_r1_pe_1");
#pragma HLS stream variable = from_r0_offset_2_to_r1_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_r0_offset_2_to_r1_pe_1
  hls::stream<Data<float>> from_diff_u_pe_1_to_diff_u_offset_2("from_diff_u_pe_1_to_diff_u_offset_2");
#pragma HLS stream variable = from_diff_u_pe_1_to_diff_u_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_pe_1_to_diff_u_offset_2
  hls::stream<Data<float>> from_diff_d_offset_2_to_g_pe_1("from_diff_d_offset_2_to_g_pe_1");
#pragma HLS stream variable = from_diff_d_offset_2_to_g_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_diff_d_offset_2_to_g_pe_1
  hls::stream<Data<float>> from_u_offset_35_to_u_offset_63("from_u_offset_35_to_u_offset_63");
#pragma HLS stream variable = from_u_offset_35_to_u_offset_63 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_35_to_u_offset_63
  hls::stream<Data<float>> from_u_offset_35_to_diff_d_pe_0("from_u_offset_35_to_diff_d_pe_0");
#pragma HLS stream variable = from_u_offset_35_to_diff_d_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_35_to_diff_d_pe_0
  hls::stream<Data<float>> from_u_offset_35_to_diff_l_pe_1("from_u_offset_35_to_diff_l_pe_1");
#pragma HLS stream variable = from_u_offset_35_to_diff_l_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_35_to_diff_l_pe_1
  hls::stream<Data<float>> from_u_offset_35_to_diff_l_pe_0("from_u_offset_35_to_diff_l_pe_0");
#pragma HLS stream variable = from_u_offset_35_to_diff_l_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_35_to_diff_l_pe_0
  hls::stream<Data<float>> from_u_offset_35_to_diff_r_pe_0("from_u_offset_35_to_diff_r_pe_0");
#pragma HLS stream variable = from_u_offset_35_to_diff_r_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_u_offset_35_to_diff_r_pe_0
  hls::stream<Data<float>> from_u_offset_35_to_diff_u_pe_0("from_u_offset_35_to_diff_u_pe_0");
#pragma HLS stream variable = from_u_offset_35_to_diff_u_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_u_offset_35_to_diff_u_pe_0
  hls::stream<Data<float>> from_u_offset_35_to_output_pe_0("from_u_offset_35_to_output_pe_0");
#pragma HLS stream variable = from_u_offset_35_to_output_pe_0 depth = 17
#pragma HLS aggregate bit variable = from_u_offset_35_to_output_pe_0
  hls::stream<Data<float>> from_diff_d_pe_0_to_diff_d_offset_3("from_diff_d_pe_0_to_diff_d_offset_3");
#pragma HLS stream variable = from_diff_d_pe_0_to_diff_d_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_diff_d_pe_0_to_diff_d_offset_3
  hls::stream<Data<float>> from_diff_l_pe_1_to_diff_l_offset_2("from_diff_l_pe_1_to_diff_l_offset_2");
#pragma HLS stream variable = from_diff_l_pe_1_to_diff_l_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_l_pe_1_to_diff_l_offset_2
  hls::stream<Data<float>> from_diff_r_pe_0_to_diff_r_offset_3("from_diff_r_pe_0_to_diff_r_offset_3");
#pragma HLS stream variable = from_diff_r_pe_0_to_diff_r_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_diff_r_pe_0_to_diff_r_offset_3
  hls::stream<Data<float>> from_diff_d_offset_3_to_g_pe_0("from_diff_d_offset_3_to_g_pe_0");
#pragma HLS stream variable = from_diff_d_offset_3_to_g_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_diff_d_offset_3_to_g_pe_0
  hls::stream<Data<float>> from_r1_pe_2_to_r1_offset_1("from_r1_pe_2_to_r1_offset_1");
#pragma HLS stream variable = from_r1_pe_2_to_r1_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_r1_pe_2_to_r1_offset_1
  hls::stream<Data<float>> from_r1_pe_1_to_r1_offset_2("from_r1_pe_1_to_r1_offset_2");
#pragma HLS stream variable = from_r1_pe_1_to_r1_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_r1_pe_1_to_r1_offset_2
  hls::stream<Data<float>> from_u_offset_64_to_u_offset_68("from_u_offset_64_to_u_offset_68");
#pragma HLS stream variable = from_u_offset_64_to_u_offset_68 depth = 9
#pragma HLS aggregate bit variable = from_u_offset_64_to_u_offset_68
  hls::stream<Data<float>> from_u_offset_64_to_diff_u_pe_3("from_u_offset_64_to_diff_u_pe_3");
#pragma HLS stream variable = from_u_offset_64_to_diff_u_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_64_to_diff_u_pe_3
  hls::stream<Data<float>> from_u_offset_64_to_r0_pe_3("from_u_offset_64_to_r0_pe_3");
#pragma HLS stream variable = from_u_offset_64_to_r0_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_64_to_r0_pe_3
  hls::stream<Data<float>> from_u_offset_64_to_output_pe_3("from_u_offset_64_to_output_pe_3");
#pragma HLS stream variable = from_u_offset_64_to_output_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_64_to_output_pe_3
  hls::stream<Data<float>> from_u_offset_64_to_output_pe_2("from_u_offset_64_to_output_pe_2");
#pragma HLS stream variable = from_u_offset_64_to_output_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_u_offset_64_to_output_pe_2
  hls::stream<Data<float>> from_r0_pe_3_to_r0_offset_0("from_r0_pe_3_to_r0_offset_0");
#pragma HLS stream variable = from_r0_pe_3_to_r0_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_r0_pe_3_to_r0_offset_0
  hls::stream<Data<float>> from_r0_offset_0_to_r1_pe_3("from_r0_offset_0_to_r1_pe_3");
#pragma HLS stream variable = from_r0_offset_0_to_r1_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_r0_offset_0_to_r1_pe_3
  hls::stream<Data<float>> from_diff_u_pe_3_to_diff_u_offset_0("from_diff_u_pe_3_to_diff_u_offset_0");
#pragma HLS stream variable = from_diff_u_pe_3_to_diff_u_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_pe_3_to_diff_u_offset_0
  hls::stream<Data<float>> from_r1_pe_3_to_r1_offset_0("from_r1_pe_3_to_r1_offset_0");
#pragma HLS stream variable = from_r1_pe_3_to_r1_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_r1_pe_3_to_r1_offset_0
  hls::stream<Data<float>> from_diff_l_pe_0_to_diff_l_offset_3("from_diff_l_pe_0_to_diff_l_offset_3");
#pragma HLS stream variable = from_diff_l_pe_0_to_diff_l_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_diff_l_pe_0_to_diff_l_offset_3
  hls::stream<Data<float>> from_diff_l_offset_0_to_g_pe_3("from_diff_l_offset_0_to_g_pe_3");
#pragma HLS stream variable = from_diff_l_offset_0_to_g_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_diff_l_offset_0_to_g_pe_3
  hls::stream<Data<float>> from_diff_r_offset_0_to_g_pe_3("from_diff_r_offset_0_to_g_pe_3");
#pragma HLS stream variable = from_diff_r_offset_0_to_g_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_diff_r_offset_0_to_g_pe_3
  hls::stream<Data<float>> from_diff_r_offset_1_to_g_pe_2("from_diff_r_offset_1_to_g_pe_2");
#pragma HLS stream variable = from_diff_r_offset_1_to_g_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_diff_r_offset_1_to_g_pe_2
  hls::stream<Data<float>> from_diff_u_offset_0_to_g_pe_3("from_diff_u_offset_0_to_g_pe_3");
#pragma HLS stream variable = from_diff_u_offset_0_to_g_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_offset_0_to_g_pe_3
  hls::stream<Data<float>> from_g_pe_3_to_g_offset_0("from_g_pe_3_to_g_offset_0");
#pragma HLS stream variable = from_g_pe_3_to_g_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_g_pe_3_to_g_offset_0
  hls::stream<Data<float>> from_u_offset_97_to_output_pe_2("from_u_offset_97_to_output_pe_2");
#pragma HLS stream variable = from_u_offset_97_to_output_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_u_offset_97_to_output_pe_2
  hls::stream<Data<float>> from_diff_l_offset_1_to_g_pe_2("from_diff_l_offset_1_to_g_pe_2");
#pragma HLS stream variable = from_diff_l_offset_1_to_g_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_l_offset_1_to_g_pe_2
  hls::stream<Data<float>> from_diff_r_offset_2_to_g_pe_1("from_diff_r_offset_2_to_g_pe_1");
#pragma HLS stream variable = from_diff_r_offset_2_to_g_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_diff_r_offset_2_to_g_pe_1
  hls::stream<Data<float>> from_diff_u_offset_1_to_g_pe_2("from_diff_u_offset_1_to_g_pe_2");
#pragma HLS stream variable = from_diff_u_offset_1_to_g_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_offset_1_to_g_pe_2
  hls::stream<Data<float>> from_g_pe_2_to_g_offset_1("from_g_pe_2_to_g_offset_1");
#pragma HLS stream variable = from_g_pe_2_to_g_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_g_pe_2_to_g_offset_1
  hls::stream<Data<float>> from_u_offset_98_to_output_pe_1("from_u_offset_98_to_output_pe_1");
#pragma HLS stream variable = from_u_offset_98_to_output_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_u_offset_98_to_output_pe_1
  hls::stream<Data<float>> from_diff_l_offset_2_to_g_pe_1("from_diff_l_offset_2_to_g_pe_1");
#pragma HLS stream variable = from_diff_l_offset_2_to_g_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_diff_l_offset_2_to_g_pe_1
  hls::stream<Data<float>> from_diff_r_offset_3_to_g_pe_0("from_diff_r_offset_3_to_g_pe_0");
#pragma HLS stream variable = from_diff_r_offset_3_to_g_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_diff_r_offset_3_to_g_pe_0
  hls::stream<Data<float>> from_diff_u_offset_2_to_g_pe_1("from_diff_u_offset_2_to_g_pe_1");
#pragma HLS stream variable = from_diff_u_offset_2_to_g_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_offset_2_to_g_pe_1
  hls::stream<Data<float>> from_g_pe_1_to_g_offset_2("from_g_pe_1_to_g_offset_2");
#pragma HLS stream variable = from_g_pe_1_to_g_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_g_pe_1_to_g_offset_2
  hls::stream<Data<float>> from_u_offset_63_to_u_offset_67("from_u_offset_63_to_u_offset_67");
#pragma HLS stream variable = from_u_offset_63_to_u_offset_67 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_63_to_u_offset_67
  hls::stream<Data<float>> from_u_offset_63_to_output_pe_3("from_u_offset_63_to_output_pe_3");
#pragma HLS stream variable = from_u_offset_63_to_output_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_63_to_output_pe_3
  hls::stream<Data<float>> from_r1_offset_0_to_output_pe_3("from_r1_offset_0_to_output_pe_3");
#pragma HLS stream variable = from_r1_offset_0_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_r1_offset_0_to_output_pe_3
  hls::stream<Data<float>> from_r1_offset_1_to_output_pe_2("from_r1_offset_1_to_output_pe_2");
#pragma HLS stream variable = from_r1_offset_1_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_r1_offset_1_to_output_pe_2
  hls::stream<Data<float>> from_r1_offset_2_to_output_pe_1("from_r1_offset_2_to_output_pe_1");
#pragma HLS stream variable = from_r1_offset_2_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_r1_offset_2_to_output_pe_1
  hls::stream<Data<float>> from_u_offset_68_to_u_offset_96("from_u_offset_68_to_u_offset_96");
#pragma HLS stream variable = from_u_offset_68_to_u_offset_96 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_68_to_u_offset_96
  hls::stream<Data<float>> from_u_offset_68_to_output_pe_0("from_u_offset_68_to_output_pe_0");
#pragma HLS stream variable = from_u_offset_68_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_u_offset_68_to_output_pe_0
  hls::stream<Data<float>> from_diff_l_offset_3_to_g_pe_0("from_diff_l_offset_3_to_g_pe_0");
#pragma HLS stream variable = from_diff_l_offset_3_to_g_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_diff_l_offset_3_to_g_pe_0
  hls::stream<Data<float>> from_g_offset_0_to_g_offset_32("from_g_offset_0_to_g_offset_32");
#pragma HLS stream variable = from_g_offset_0_to_g_offset_32 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_0_to_g_offset_32
  hls::stream<Data<float>> from_g_offset_0_to_output_pe_3("from_g_offset_0_to_output_pe_3");
#pragma HLS stream variable = from_g_offset_0_to_output_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_g_offset_0_to_output_pe_3
  hls::stream<Data<float>> from_g_offset_1_to_g_offset_33("from_g_offset_1_to_g_offset_33");
#pragma HLS stream variable = from_g_offset_1_to_g_offset_33 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_1_to_g_offset_33
  hls::stream<Data<float>> from_g_offset_1_to_output_pe_2("from_g_offset_1_to_output_pe_2");
#pragma HLS stream variable = from_g_offset_1_to_output_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_1_to_output_pe_2
  hls::stream<Data<float>> from_g_offset_2_to_g_offset_34("from_g_offset_2_to_g_offset_34");
#pragma HLS stream variable = from_g_offset_2_to_g_offset_34 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_2_to_g_offset_34
  hls::stream<Data<float>> from_g_offset_2_to_output_pe_1("from_g_offset_2_to_output_pe_1");
#pragma HLS stream variable = from_g_offset_2_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_2_to_output_pe_1
  hls::stream<Data<float>> from_u_offset_67_to_u_offset_99("from_u_offset_67_to_u_offset_99");
#pragma HLS stream variable = from_u_offset_67_to_u_offset_99 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_67_to_u_offset_99
  hls::stream<Data<float>> from_u_offset_67_to_diff_u_pe_0("from_u_offset_67_to_diff_u_pe_0");
#pragma HLS stream variable = from_u_offset_67_to_diff_u_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_67_to_diff_u_pe_0
  hls::stream<Data<float>> from_u_offset_67_to_r0_pe_0("from_u_offset_67_to_r0_pe_0");
#pragma HLS stream variable = from_u_offset_67_to_r0_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_67_to_r0_pe_0
  hls::stream<Data<float>> from_u_offset_67_to_output_pe_1("from_u_offset_67_to_output_pe_1");
#pragma HLS stream variable = from_u_offset_67_to_output_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_u_offset_67_to_output_pe_1
  hls::stream<Data<float>> from_u_offset_67_to_output_pe_0("from_u_offset_67_to_output_pe_0");
#pragma HLS stream variable = from_u_offset_67_to_output_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_u_offset_67_to_output_pe_0
  hls::stream<Data<float>> from_r0_pe_0_to_r0_offset_3("from_r0_pe_0_to_r0_offset_3");
#pragma HLS stream variable = from_r0_pe_0_to_r0_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_r0_pe_0_to_r0_offset_3
  hls::stream<Data<float>> from_r0_offset_3_to_r1_pe_0("from_r0_offset_3_to_r1_pe_0");
#pragma HLS stream variable = from_r0_offset_3_to_r1_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_r0_offset_3_to_r1_pe_0
  hls::stream<Data<float>> from_r1_pe_0_to_r1_offset_3("from_r1_pe_0_to_r1_offset_3");
#pragma HLS stream variable = from_r1_pe_0_to_r1_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_r1_pe_0_to_r1_offset_3
  hls::stream<Data<float>> from_diff_u_pe_0_to_diff_u_offset_3("from_diff_u_pe_0_to_diff_u_offset_3");
#pragma HLS stream variable = from_diff_u_pe_0_to_diff_u_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_pe_0_to_diff_u_offset_3
  hls::stream<Data<float>> from_r1_offset_3_to_output_pe_0("from_r1_offset_3_to_output_pe_0");
#pragma HLS stream variable = from_r1_offset_3_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_r1_offset_3_to_output_pe_0
  hls::stream<Data<float>> from_diff_u_offset_3_to_g_pe_0("from_diff_u_offset_3_to_g_pe_0");
#pragma HLS stream variable = from_diff_u_offset_3_to_g_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_diff_u_offset_3_to_g_pe_0
  hls::stream<Data<float>> from_g_pe_0_to_g_offset_3("from_g_pe_0_to_g_offset_3");
#pragma HLS stream variable = from_g_pe_0_to_g_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_g_pe_0_to_g_offset_3
  hls::stream<Data<float>> from_g_offset_3_to_g_offset_31("from_g_offset_3_to_g_offset_31");
#pragma HLS stream variable = from_g_offset_3_to_g_offset_31 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_3_to_g_offset_31
  hls::stream<Data<float>> from_g_offset_3_to_output_pe_0("from_g_offset_3_to_output_pe_0");
#pragma HLS stream variable = from_g_offset_3_to_output_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_g_offset_3_to_output_pe_0
  hls::stream<Data<float>> from_u_offset_96_to_output_pe_3("from_u_offset_96_to_output_pe_3");
#pragma HLS stream variable = from_u_offset_96_to_output_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_u_offset_96_to_output_pe_3
  hls::stream<Data<float>> from_g_offset_32_to_g_offset_36("from_g_offset_32_to_g_offset_36");
#pragma HLS stream variable = from_g_offset_32_to_g_offset_36 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_32_to_g_offset_36
  hls::stream<Data<float>> from_g_offset_32_to_output_pe_2("from_g_offset_32_to_output_pe_2");
#pragma HLS stream variable = from_g_offset_32_to_output_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_g_offset_32_to_output_pe_2
  hls::stream<Data<float>> from_g_offset_33_to_g_offset_65("from_g_offset_33_to_g_offset_65");
#pragma HLS stream variable = from_g_offset_33_to_g_offset_65 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_33_to_g_offset_65
  hls::stream<Data<float>> from_g_offset_33_to_output_pe_3("from_g_offset_33_to_output_pe_3");
#pragma HLS stream variable = from_g_offset_33_to_output_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_33_to_output_pe_3
  hls::stream<Data<float>> from_g_offset_33_to_output_pe_1("from_g_offset_33_to_output_pe_1");
#pragma HLS stream variable = from_g_offset_33_to_output_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_33_to_output_pe_1
  hls::stream<Data<float>> from_g_offset_34_to_g_offset_66("from_g_offset_34_to_g_offset_66");
#pragma HLS stream variable = from_g_offset_34_to_g_offset_66 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_34_to_g_offset_66
  hls::stream<Data<float>> from_g_offset_34_to_output_pe_2("from_g_offset_34_to_output_pe_2");
#pragma HLS stream variable = from_g_offset_34_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_34_to_output_pe_2
  hls::stream<Data<float>> from_g_offset_34_to_output_pe_0("from_g_offset_34_to_output_pe_0");
#pragma HLS stream variable = from_g_offset_34_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_34_to_output_pe_0
  hls::stream<Data<float>> from_u_offset_99_to_output_pe_0("from_u_offset_99_to_output_pe_0");
#pragma HLS stream variable = from_u_offset_99_to_output_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_u_offset_99_to_output_pe_0
  hls::stream<Data<float>> from_g_offset_31_to_g_offset_35("from_g_offset_31_to_g_offset_35");
#pragma HLS stream variable = from_g_offset_31_to_g_offset_35 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_31_to_g_offset_35
  hls::stream<Data<float>> from_g_offset_31_to_output_pe_3("from_g_offset_31_to_output_pe_3");
#pragma HLS stream variable = from_g_offset_31_to_output_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_g_offset_31_to_output_pe_3
  hls::stream<Data<float>> from_g_offset_36_to_g_offset_64("from_g_offset_36_to_g_offset_64");
#pragma HLS stream variable = from_g_offset_36_to_g_offset_64 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_36_to_g_offset_64
  hls::stream<Data<float>> from_g_offset_36_to_output_pe_0("from_g_offset_36_to_output_pe_0");
#pragma HLS stream variable = from_g_offset_36_to_output_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_g_offset_36_to_output_pe_0
  hls::stream<Data<float>> from_g_offset_65_to_output_pe_2("from_g_offset_65_to_output_pe_2");
#pragma HLS stream variable = from_g_offset_65_to_output_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_65_to_output_pe_2
  hls::stream<Data<float>> from_output_pe_2_to_output_bank_2("from_output_pe_2_to_output_bank_2");
#pragma HLS stream variable = from_output_pe_2_to_output_bank_2 depth = 7
#pragma HLS aggregate bit variable = from_output_pe_2_to_output_bank_2
  hls::stream<Data<float>> from_g_offset_66_to_output_pe_1("from_g_offset_66_to_output_pe_1");
#pragma HLS stream variable = from_g_offset_66_to_output_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_66_to_output_pe_1
  hls::stream<Data<float>> from_g_offset_35_to_g_offset_67("from_g_offset_35_to_g_offset_67");
#pragma HLS stream variable = from_g_offset_35_to_g_offset_67 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_35_to_g_offset_67
  hls::stream<Data<float>> from_g_offset_35_to_output_pe_1("from_g_offset_35_to_output_pe_1");
#pragma HLS stream variable = from_g_offset_35_to_output_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_35_to_output_pe_1
  hls::stream<Data<float>> from_output_pe_1_to_output_bank_2("from_output_pe_1_to_output_bank_2");
#pragma HLS stream variable = from_output_pe_1_to_output_bank_2 depth = 5
#pragma HLS aggregate bit variable = from_output_pe_1_to_output_bank_2
  hls::stream<Data<float>> from_g_offset_64_to_output_pe_3("from_g_offset_64_to_output_pe_3");
#pragma HLS stream variable = from_g_offset_64_to_output_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_64_to_output_pe_3
  hls::stream<Data<float>> from_output_pe_3_to_output_bank_2("from_output_pe_3_to_output_bank_2");
#pragma HLS stream variable = from_output_pe_3_to_output_bank_2 depth = 5
#pragma HLS aggregate bit variable = from_output_pe_3_to_output_bank_2
  hls::stream<Data<float>> from_g_offset_67_to_output_pe_0("from_g_offset_67_to_output_pe_0");
#pragma HLS stream variable = from_g_offset_67_to_output_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_g_offset_67_to_output_pe_0
  hls::stream<Data<float>> from_output_pe_0_to_output_bank_2("from_output_pe_0_to_output_bank_2");
#pragma HLS stream variable = from_output_pe_0_to_output_bank_2 depth = 3
#pragma HLS aggregate bit variable = from_output_pe_0_to_output_bank_2

#pragma HLS dataflow
  BurstRead(bank_0_f_buf, bank_0_f, coalesced_data_num);
  BurstRead(bank_1_u_buf, bank_1_u, coalesced_data_num);
  Module0Func(
      /*output*/ from_f_bank_0_to_f_offset_0, 
      /*output*/ from_f_bank_0_to_f_offset_1, 
      /*output*/ from_f_bank_0_to_f_offset_2, 
      /*output*/ from_f_bank_0_to_f_offset_3, 
      /* input*/ bank_0_f_buf);
  Module1Func(
      /*output*/ from_u_bank_1_to_u_offset_0, 
      /*output*/ from_u_bank_1_to_u_offset_1, 
      /*output*/ from_u_bank_1_to_u_offset_2, 
      /*output*/ from_u_bank_1_to_u_offset_3, 
      /* input*/ bank_1_u_buf);
  Module2Func(
      /*output*/ from_f_offset_0_to_r0_pe_3, 
      /*output*/ from_f_offset_0_to_output_pe_3, 
      /* input*/ from_f_bank_0_to_f_offset_0);
  Module2Func(
      /*output*/ from_f_offset_1_to_r0_pe_2, 
      /*output*/ from_f_offset_1_to_output_pe_2, 
      /* input*/ from_f_bank_0_to_f_offset_1);
  Module2Func(
      /*output*/ from_f_offset_2_to_r0_pe_1, 
      /*output*/ from_f_offset_2_to_output_pe_1, 
      /* input*/ from_f_bank_0_to_f_offset_2);
  Module2Func(
      /*output*/ from_f_offset_3_to_r0_pe_0, 
      /*output*/ from_f_offset_3_to_output_pe_0, 
      /* input*/ from_f_bank_0_to_f_offset_3);
  Module2Func(
      /*output*/ from_u_offset_0_to_u_offset_32, 
      /*output*/ from_u_offset_0_to_diff_d_pe_3, 
      /* input*/ from_u_bank_1_to_u_offset_0);
  Module2Func(
      /*output*/ from_u_offset_1_to_u_offset_33, 
      /*output*/ from_u_offset_1_to_diff_d_pe_2, 
      /* input*/ from_u_bank_1_to_u_offset_1);
  Module2Func(
      /*output*/ from_u_offset_2_to_u_offset_34, 
      /*output*/ from_u_offset_2_to_diff_d_pe_1, 
      /* input*/ from_u_bank_1_to_u_offset_2);
  Module2Func(
      /*output*/ from_u_offset_3_to_u_offset_31, 
      /*output*/ from_u_offset_3_to_diff_d_pe_0, 
      /* input*/ from_u_bank_1_to_u_offset_3);
  Module3Func(
      /*output*/ from_u_offset_32_to_u_offset_36, 
      /*output*/ from_u_offset_32_to_diff_d_pe_3, 
      /*output*/ from_u_offset_32_to_diff_l_pe_3, 
      /*output*/ from_u_offset_32_to_diff_r_pe_3, 
      /*output*/ from_u_offset_32_to_diff_r_pe_2, 
      /*output*/ from_u_offset_32_to_diff_u_pe_3, 
      /*output*/ from_u_offset_32_to_output_pe_3, 
      /* input*/ from_u_offset_0_to_u_offset_32);
  Module4Func(
      /*output*/ from_diff_d_pe_3_to_diff_d_offset_0, 
      /* input*/ from_u_offset_32_to_diff_d_pe_3, 
      /* input*/ from_u_offset_0_to_diff_d_pe_3);
  Module5Func(
      /*output*/ from_u_offset_33_to_u_offset_65, 
      /*output*/ from_u_offset_33_to_diff_d_pe_2, 
      /*output*/ from_u_offset_33_to_diff_l_pe_3, 
      /*output*/ from_u_offset_33_to_diff_l_pe_2, 
      /*output*/ from_u_offset_33_to_diff_r_pe_2, 
      /*output*/ from_u_offset_33_to_diff_r_pe_1, 
      /*output*/ from_u_offset_33_to_diff_u_pe_2, 
      /*output*/ from_u_offset_33_to_output_pe_2, 
      /* input*/ from_u_offset_1_to_u_offset_33);
  Module4Func(
      /*output*/ from_diff_d_pe_2_to_diff_d_offset_1, 
      /* input*/ from_u_offset_33_to_diff_d_pe_2, 
      /* input*/ from_u_offset_1_to_diff_d_pe_2);
  Module5Func(
      /*output*/ from_u_offset_34_to_u_offset_66, 
      /*output*/ from_u_offset_34_to_diff_d_pe_1, 
      /*output*/ from_u_offset_34_to_diff_l_pe_2, 
      /*output*/ from_u_offset_34_to_diff_l_pe_1, 
      /*output*/ from_u_offset_34_to_diff_r_pe_1, 
      /*output*/ from_u_offset_34_to_diff_r_pe_0, 
      /*output*/ from_u_offset_34_to_diff_u_pe_1, 
      /*output*/ from_u_offset_34_to_output_pe_1, 
      /* input*/ from_u_offset_2_to_u_offset_34);
  Module4Func(
      /*output*/ from_diff_d_pe_1_to_diff_d_offset_2, 
      /* input*/ from_u_offset_34_to_diff_d_pe_1, 
      /* input*/ from_u_offset_2_to_diff_d_pe_1);
  Module6Func(
      /*output*/ from_u_offset_31_to_u_offset_35, 
      /*output*/ from_u_offset_31_to_diff_r_pe_3, 
      /* input*/ from_u_offset_3_to_u_offset_31);
  Module7Func(
      /*output*/ from_u_offset_36_to_u_offset_64, 
      /*output*/ from_u_offset_36_to_diff_l_pe_0, 
      /* input*/ from_u_offset_32_to_u_offset_36);
  Module4Func(
      /*output*/ from_diff_l_pe_3_to_diff_l_offset_0, 
      /* input*/ from_u_offset_32_to_diff_l_pe_3, 
      /* input*/ from_u_offset_33_to_diff_l_pe_3);
  Module4Func(
      /*output*/ from_diff_r_pe_3_to_diff_r_offset_0, 
      /* input*/ from_u_offset_32_to_diff_r_pe_3, 
      /* input*/ from_u_offset_31_to_diff_r_pe_3);
  Module4Func(
      /*output*/ from_diff_r_pe_2_to_diff_r_offset_1, 
      /* input*/ from_u_offset_33_to_diff_r_pe_2, 
      /* input*/ from_u_offset_32_to_diff_r_pe_2);
  Module8Func(
      /*output*/ from_diff_d_offset_0_to_g_pe_3, 
      /* input*/ from_diff_d_pe_3_to_diff_d_offset_0);
  Module9Func(
      /*output*/ from_u_offset_65_to_u_offset_97, 
      /*output*/ from_u_offset_65_to_diff_u_pe_2, 
      /*output*/ from_u_offset_65_to_r0_pe_2, 
      /*output*/ from_u_offset_65_to_output_pe_3, 
      /*output*/ from_u_offset_65_to_output_pe_2, 
      /*output*/ from_u_offset_65_to_output_pe_1, 
      /* input*/ from_u_offset_33_to_u_offset_65);
  Module10Func(
      /*output*/ from_r0_pe_2_to_r0_offset_1, 
      /* input*/ from_u_offset_65_to_r0_pe_2, 
      /* input*/ from_f_offset_1_to_r0_pe_2);
  Module8Func(
      /*output*/ from_r0_offset_1_to_r1_pe_2, 
      /* input*/ from_r0_pe_2_to_r0_offset_1);
  Module4Func(
      /*output*/ from_diff_l_pe_2_to_diff_l_offset_1, 
      /* input*/ from_u_offset_33_to_diff_l_pe_2, 
      /* input*/ from_u_offset_34_to_diff_l_pe_2);
  Module4Func(
      /*output*/ from_diff_r_pe_1_to_diff_r_offset_2, 
      /* input*/ from_u_offset_34_to_diff_r_pe_1, 
      /* input*/ from_u_offset_33_to_diff_r_pe_1);
  Module4Func(
      /*output*/ from_diff_u_pe_2_to_diff_u_offset_1, 
      /* input*/ from_u_offset_33_to_diff_u_pe_2, 
      /* input*/ from_u_offset_65_to_diff_u_pe_2);
  Module8Func(
      /*output*/ from_diff_d_offset_1_to_g_pe_2, 
      /* input*/ from_diff_d_pe_2_to_diff_d_offset_1);
  Module9Func(
      /*output*/ from_u_offset_66_to_u_offset_98, 
      /*output*/ from_u_offset_66_to_diff_u_pe_1, 
      /*output*/ from_u_offset_66_to_r0_pe_1, 
      /*output*/ from_u_offset_66_to_output_pe_2, 
      /*output*/ from_u_offset_66_to_output_pe_1, 
      /*output*/ from_u_offset_66_to_output_pe_0, 
      /* input*/ from_u_offset_34_to_u_offset_66);
  Module10Func(
      /*output*/ from_r0_pe_1_to_r0_offset_2, 
      /* input*/ from_u_offset_66_to_r0_pe_1, 
      /* input*/ from_f_offset_2_to_r0_pe_1);
  Module8Func(
      /*output*/ from_r0_offset_2_to_r1_pe_1, 
      /* input*/ from_r0_pe_1_to_r0_offset_2);
  Module4Func(
      /*output*/ from_diff_u_pe_1_to_diff_u_offset_2, 
      /* input*/ from_u_offset_34_to_diff_u_pe_1, 
      /* input*/ from_u_offset_66_to_diff_u_pe_1);
  Module8Func(
      /*output*/ from_diff_d_offset_2_to_g_pe_1, 
      /* input*/ from_diff_d_pe_1_to_diff_d_offset_2);
  Module11Func(
      /*output*/ from_u_offset_35_to_u_offset_63, 
      /*output*/ from_u_offset_35_to_diff_d_pe_0, 
      /*output*/ from_u_offset_35_to_diff_l_pe_1, 
      /*output*/ from_u_offset_35_to_diff_l_pe_0, 
      /*output*/ from_u_offset_35_to_diff_r_pe_0, 
      /*output*/ from_u_offset_35_to_diff_u_pe_0, 
      /*output*/ from_u_offset_35_to_output_pe_0, 
      /* input*/ from_u_offset_31_to_u_offset_35);
  Module4Func(
      /*output*/ from_diff_d_pe_0_to_diff_d_offset_3, 
      /* input*/ from_u_offset_35_to_diff_d_pe_0, 
      /* input*/ from_u_offset_3_to_diff_d_pe_0);
  Module4Func(
      /*output*/ from_diff_l_pe_1_to_diff_l_offset_2, 
      /* input*/ from_u_offset_34_to_diff_l_pe_1, 
      /* input*/ from_u_offset_35_to_diff_l_pe_1);
  Module4Func(
      /*output*/ from_diff_r_pe_0_to_diff_r_offset_3, 
      /* input*/ from_u_offset_35_to_diff_r_pe_0, 
      /* input*/ from_u_offset_34_to_diff_r_pe_0);
  Module8Func(
      /*output*/ from_diff_d_offset_3_to_g_pe_0, 
      /* input*/ from_diff_d_pe_0_to_diff_d_offset_3);
  Module12Func(
      /*output*/ from_r1_pe_2_to_r1_offset_1, 
      /* input*/ from_r0_offset_1_to_r1_pe_2);
  Module12Func(
      /*output*/ from_r1_pe_1_to_r1_offset_2, 
      /* input*/ from_r0_offset_2_to_r1_pe_1);
  Module13Func(
      /*output*/ from_u_offset_64_to_u_offset_68, 
      /*output*/ from_u_offset_64_to_diff_u_pe_3, 
      /*output*/ from_u_offset_64_to_r0_pe_3, 
      /*output*/ from_u_offset_64_to_output_pe_3, 
      /*output*/ from_u_offset_64_to_output_pe_2, 
      /* input*/ from_u_offset_36_to_u_offset_64);
  Module10Func(
      /*output*/ from_r0_pe_3_to_r0_offset_0, 
      /* input*/ from_u_offset_64_to_r0_pe_3, 
      /* input*/ from_f_offset_0_to_r0_pe_3);
  Module8Func(
      /*output*/ from_r0_offset_0_to_r1_pe_3, 
      /* input*/ from_r0_pe_3_to_r0_offset_0);
  Module4Func(
      /*output*/ from_diff_u_pe_3_to_diff_u_offset_0, 
      /* input*/ from_u_offset_32_to_diff_u_pe_3, 
      /* input*/ from_u_offset_64_to_diff_u_pe_3);
  Module12Func(
      /*output*/ from_r1_pe_3_to_r1_offset_0, 
      /* input*/ from_r0_offset_0_to_r1_pe_3);
  Module4Func(
      /*output*/ from_diff_l_pe_0_to_diff_l_offset_3, 
      /* input*/ from_u_offset_35_to_diff_l_pe_0, 
      /* input*/ from_u_offset_36_to_diff_l_pe_0);
  Module8Func(
      /*output*/ from_diff_l_offset_0_to_g_pe_3, 
      /* input*/ from_diff_l_pe_3_to_diff_l_offset_0);
  Module8Func(
      /*output*/ from_diff_r_offset_0_to_g_pe_3, 
      /* input*/ from_diff_r_pe_3_to_diff_r_offset_0);
  Module8Func(
      /*output*/ from_diff_r_offset_1_to_g_pe_2, 
      /* input*/ from_diff_r_pe_2_to_diff_r_offset_1);
  Module8Func(
      /*output*/ from_diff_u_offset_0_to_g_pe_3, 
      /* input*/ from_diff_u_pe_3_to_diff_u_offset_0);
  Module14Func(
      /*output*/ from_g_pe_3_to_g_offset_0, 
      /* input*/ from_diff_u_offset_0_to_g_pe_3, 
      /* input*/ from_diff_d_offset_0_to_g_pe_3, 
      /* input*/ from_diff_l_offset_0_to_g_pe_3, 
      /* input*/ from_diff_r_offset_0_to_g_pe_3);
  Module15Func(
      /*output*/ from_u_offset_97_to_output_pe_2, 
      /* input*/ from_u_offset_65_to_u_offset_97);
  Module8Func(
      /*output*/ from_diff_l_offset_1_to_g_pe_2, 
      /* input*/ from_diff_l_pe_2_to_diff_l_offset_1);
  Module8Func(
      /*output*/ from_diff_r_offset_2_to_g_pe_1, 
      /* input*/ from_diff_r_pe_1_to_diff_r_offset_2);
  Module8Func(
      /*output*/ from_diff_u_offset_1_to_g_pe_2, 
      /* input*/ from_diff_u_pe_2_to_diff_u_offset_1);
  Module14Func(
      /*output*/ from_g_pe_2_to_g_offset_1, 
      /* input*/ from_diff_u_offset_1_to_g_pe_2, 
      /* input*/ from_diff_d_offset_1_to_g_pe_2, 
      /* input*/ from_diff_l_offset_1_to_g_pe_2, 
      /* input*/ from_diff_r_offset_1_to_g_pe_2);
  Module15Func(
      /*output*/ from_u_offset_98_to_output_pe_1, 
      /* input*/ from_u_offset_66_to_u_offset_98);
  Module8Func(
      /*output*/ from_diff_l_offset_2_to_g_pe_1, 
      /* input*/ from_diff_l_pe_1_to_diff_l_offset_2);
  Module8Func(
      /*output*/ from_diff_r_offset_3_to_g_pe_0, 
      /* input*/ from_diff_r_pe_0_to_diff_r_offset_3);
  Module8Func(
      /*output*/ from_diff_u_offset_2_to_g_pe_1, 
      /* input*/ from_diff_u_pe_1_to_diff_u_offset_2);
  Module14Func(
      /*output*/ from_g_pe_1_to_g_offset_2, 
      /* input*/ from_diff_u_offset_2_to_g_pe_1, 
      /* input*/ from_diff_d_offset_2_to_g_pe_1, 
      /* input*/ from_diff_l_offset_2_to_g_pe_1, 
      /* input*/ from_diff_r_offset_2_to_g_pe_1);
  Module6Func(
      /*output*/ from_u_offset_63_to_u_offset_67, 
      /*output*/ from_u_offset_63_to_output_pe_3, 
      /* input*/ from_u_offset_35_to_u_offset_63);
  Module8Func(
      /*output*/ from_r1_offset_0_to_output_pe_3, 
      /* input*/ from_r1_pe_3_to_r1_offset_0);
  Module8Func(
      /*output*/ from_r1_offset_1_to_output_pe_2, 
      /* input*/ from_r1_pe_2_to_r1_offset_1);
  Module8Func(
      /*output*/ from_r1_offset_2_to_output_pe_1, 
      /* input*/ from_r1_pe_1_to_r1_offset_2);
  Module7Func(
      /*output*/ from_u_offset_68_to_u_offset_96, 
      /*output*/ from_u_offset_68_to_output_pe_0, 
      /* input*/ from_u_offset_64_to_u_offset_68);
  Module8Func(
      /*output*/ from_diff_l_offset_3_to_g_pe_0, 
      /* input*/ from_diff_l_pe_0_to_diff_l_offset_3);
  Module2Func(
      /*output*/ from_g_offset_0_to_g_offset_32, 
      /*output*/ from_g_offset_0_to_output_pe_3, 
      /* input*/ from_g_pe_3_to_g_offset_0);
  Module2Func(
      /*output*/ from_g_offset_1_to_g_offset_33, 
      /*output*/ from_g_offset_1_to_output_pe_2, 
      /* input*/ from_g_pe_2_to_g_offset_1);
  Module2Func(
      /*output*/ from_g_offset_2_to_g_offset_34, 
      /*output*/ from_g_offset_2_to_output_pe_1, 
      /* input*/ from_g_pe_1_to_g_offset_2);
  Module16Func(
      /*output*/ from_u_offset_67_to_u_offset_99, 
      /*output*/ from_u_offset_67_to_diff_u_pe_0, 
      /*output*/ from_u_offset_67_to_r0_pe_0, 
      /*output*/ from_u_offset_67_to_output_pe_1, 
      /*output*/ from_u_offset_67_to_output_pe_0, 
      /* input*/ from_u_offset_63_to_u_offset_67);
  Module10Func(
      /*output*/ from_r0_pe_0_to_r0_offset_3, 
      /* input*/ from_u_offset_67_to_r0_pe_0, 
      /* input*/ from_f_offset_3_to_r0_pe_0);
  Module8Func(
      /*output*/ from_r0_offset_3_to_r1_pe_0, 
      /* input*/ from_r0_pe_0_to_r0_offset_3);
  Module12Func(
      /*output*/ from_r1_pe_0_to_r1_offset_3, 
      /* input*/ from_r0_offset_3_to_r1_pe_0);
  Module4Func(
      /*output*/ from_diff_u_pe_0_to_diff_u_offset_3, 
      /* input*/ from_u_offset_35_to_diff_u_pe_0, 
      /* input*/ from_u_offset_67_to_diff_u_pe_0);
  Module8Func(
      /*output*/ from_r1_offset_3_to_output_pe_0, 
      /* input*/ from_r1_pe_0_to_r1_offset_3);
  Module8Func(
      /*output*/ from_diff_u_offset_3_to_g_pe_0, 
      /* input*/ from_diff_u_pe_0_to_diff_u_offset_3);
  Module14Func(
      /*output*/ from_g_pe_0_to_g_offset_3, 
      /* input*/ from_diff_u_offset_3_to_g_pe_0, 
      /* input*/ from_diff_d_offset_3_to_g_pe_0, 
      /* input*/ from_diff_l_offset_3_to_g_pe_0, 
      /* input*/ from_diff_r_offset_3_to_g_pe_0);
  Module2Func(
      /*output*/ from_g_offset_3_to_g_offset_31, 
      /*output*/ from_g_offset_3_to_output_pe_0, 
      /* input*/ from_g_pe_0_to_g_offset_3);
  Module17Func(
      /*output*/ from_u_offset_96_to_output_pe_3, 
      /* input*/ from_u_offset_68_to_u_offset_96);
  Module18Func(
      /*output*/ from_g_offset_32_to_g_offset_36, 
      /*output*/ from_g_offset_32_to_output_pe_2, 
      /* input*/ from_g_offset_0_to_g_offset_32);
  Module19Func(
      /*output*/ from_g_offset_33_to_g_offset_65, 
      /*output*/ from_g_offset_33_to_output_pe_3, 
      /*output*/ from_g_offset_33_to_output_pe_1, 
      /* input*/ from_g_offset_1_to_g_offset_33);
  Module19Func(
      /*output*/ from_g_offset_34_to_g_offset_66, 
      /*output*/ from_g_offset_34_to_output_pe_2, 
      /*output*/ from_g_offset_34_to_output_pe_0, 
      /* input*/ from_g_offset_2_to_g_offset_34);
  Module15Func(
      /*output*/ from_u_offset_99_to_output_pe_0, 
      /* input*/ from_u_offset_67_to_u_offset_99);
  Module6Func(
      /*output*/ from_g_offset_31_to_g_offset_35, 
      /*output*/ from_g_offset_31_to_output_pe_3, 
      /* input*/ from_g_offset_3_to_g_offset_31);
  Module7Func(
      /*output*/ from_g_offset_36_to_g_offset_64, 
      /*output*/ from_g_offset_36_to_output_pe_0, 
      /* input*/ from_g_offset_32_to_g_offset_36);
  Module15Func(
      /*output*/ from_g_offset_65_to_output_pe_2, 
      /* input*/ from_g_offset_33_to_g_offset_65);
  Module20Func(
      /*output*/ from_output_pe_2_to_output_bank_2, 
      /* input*/ from_u_offset_65_to_output_pe_2, 
      /* input*/ from_u_offset_33_to_output_pe_2, 
      /* input*/ from_g_offset_1_to_output_pe_2, 
      /* input*/ from_u_offset_97_to_output_pe_2, 
      /* input*/ from_g_offset_65_to_output_pe_2, 
      /* input*/ from_u_offset_66_to_output_pe_2, 
      /* input*/ from_g_offset_34_to_output_pe_2, 
      /* input*/ from_u_offset_64_to_output_pe_2, 
      /* input*/ from_g_offset_32_to_output_pe_2, 
      /* input*/ from_f_offset_1_to_output_pe_2, 
      /* input*/ from_r1_offset_1_to_output_pe_2);
  Module15Func(
      /*output*/ from_g_offset_66_to_output_pe_1, 
      /* input*/ from_g_offset_34_to_g_offset_66);
  Module7Func(
      /*output*/ from_g_offset_35_to_g_offset_67, 
      /*output*/ from_g_offset_35_to_output_pe_1, 
      /* input*/ from_g_offset_31_to_g_offset_35);
  Module20Func(
      /*output*/ from_output_pe_1_to_output_bank_2, 
      /* input*/ from_u_offset_66_to_output_pe_1, 
      /* input*/ from_u_offset_34_to_output_pe_1, 
      /* input*/ from_g_offset_2_to_output_pe_1, 
      /* input*/ from_u_offset_98_to_output_pe_1, 
      /* input*/ from_g_offset_66_to_output_pe_1, 
      /* input*/ from_u_offset_67_to_output_pe_1, 
      /* input*/ from_g_offset_35_to_output_pe_1, 
      /* input*/ from_u_offset_65_to_output_pe_1, 
      /* input*/ from_g_offset_33_to_output_pe_1, 
      /* input*/ from_f_offset_2_to_output_pe_1, 
      /* input*/ from_r1_offset_2_to_output_pe_1);
  Module17Func(
      /*output*/ from_g_offset_64_to_output_pe_3, 
      /* input*/ from_g_offset_36_to_g_offset_64);
  Module20Func(
      /*output*/ from_output_pe_3_to_output_bank_2, 
      /* input*/ from_u_offset_64_to_output_pe_3, 
      /* input*/ from_u_offset_32_to_output_pe_3, 
      /* input*/ from_g_offset_0_to_output_pe_3, 
      /* input*/ from_u_offset_96_to_output_pe_3, 
      /* input*/ from_g_offset_64_to_output_pe_3, 
      /* input*/ from_u_offset_65_to_output_pe_3, 
      /* input*/ from_g_offset_33_to_output_pe_3, 
      /* input*/ from_u_offset_63_to_output_pe_3, 
      /* input*/ from_g_offset_31_to_output_pe_3, 
      /* input*/ from_f_offset_0_to_output_pe_3, 
      /* input*/ from_r1_offset_0_to_output_pe_3);
  Module15Func(
      /*output*/ from_g_offset_67_to_output_pe_0, 
      /* input*/ from_g_offset_35_to_g_offset_67);
  Module20Func(
      /*output*/ from_output_pe_0_to_output_bank_2, 
      /* input*/ from_u_offset_67_to_output_pe_0, 
      /* input*/ from_u_offset_35_to_output_pe_0, 
      /* input*/ from_g_offset_3_to_output_pe_0, 
      /* input*/ from_u_offset_99_to_output_pe_0, 
      /* input*/ from_g_offset_67_to_output_pe_0, 
      /* input*/ from_u_offset_68_to_output_pe_0, 
      /* input*/ from_g_offset_36_to_output_pe_0, 
      /* input*/ from_u_offset_66_to_output_pe_0, 
      /* input*/ from_g_offset_34_to_output_pe_0, 
      /* input*/ from_f_offset_3_to_output_pe_0, 
      /* input*/ from_r1_offset_3_to_output_pe_0);
  Module21Func(
      /*output*/ bank_2_output_buf, 
      /* input*/ from_output_pe_2_to_output_bank_2, 
      /* input*/ from_output_pe_1_to_output_bank_2, 
      /* input*/ from_output_pe_3_to_output_bank_2, 
      /* input*/ from_output_pe_0_to_output_bank_2);
  BurstWrite(bank_2_output, bank_2_output_buf, coalesced_data_num);
}

}  // extern "C"
