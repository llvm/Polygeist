// this file can be generated from the following SODA DSL
/*
kernel: sobel2d
burst width: 64
iterate: 1
unroll factor: 4
input dram 0 int16: img(32, *)
local int16: mag_x(0, 0) = (img(1, -1) - img(-1, -1)) + ((img(1, 0) - img(-1, 0)) * 3) + (img(1, 1) - img(-1, 1))
local uint16: mag_y(0, 0) = uint16((img(-1, 1) - img(-1, -1)) + ((img(0, 1) - img(0, -1)) * 3) + (img(1, 1) - img(1, -1)))
output dram 1 uint16: mag(0, 0) = uint16(65535 - ((mag_x(0, 0) * mag_x(0, 0)) + (mag_y(0, 0) * mag_y(0, 0))))
border: ignore
cluster: none
*/

// stencil window size: (3, 3)
// stencil distace: 66
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
  /* input*/ hls::stream<Data<ap_uint<64>>>& dram_img_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_img_bank_0_fifo
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
    if (!dram_img_bank_0_fifo.empty())
    {
      ap_uint<64> dram_img_bank_0_buf;
      const bool dram_img_bank_0_buf_enable = ReadData(dram_img_bank_0_buf, dram_img_bank_0_fifo);
      const bool enabled = dram_img_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_img_bank_0_buf(31, 16))), enabled);
      WriteData(fifo_st_1, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_img_bank_0_buf(63, 48))), enabled);
      WriteData(fifo_st_2, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_img_bank_0_buf(15, 0))), enabled);
      WriteData(fifo_st_3, Reinterpret<int16_t>(static_cast<ap_uint<16>>(dram_img_bank_0_buf(47, 32))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_3, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_4, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_5, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_st_4
#pragma HLS aggregate bit variable = fifo_st_5
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
      WriteData(fifo_st_2, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_3, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_4, int16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_5, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
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
module_2:
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
  } // for module_2
} // Module2Func

void Module3Func(
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
module_3:
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
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_8 = 0;
  int16_t fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const int16_t let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
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
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
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
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
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
  ap_uint<3> ptr_delay_8 = 0;
  int16_t fifo_ref_0_delayed_8_buf[8];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_8_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_7:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_8_buf distance = 8
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_8 = fifo_ref_0_delayed_8_buf[ptr_delay_8];;
      const int16_t let_0 = fifo_ref_0_delayed_8;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      WriteData(fifo_st_3, int16_t(let_0), enabled);
      WriteData(fifo_st_4, int16_t(let_0), enabled);
      fifo_ref_0_delayed_8_buf[ptr_delay_8] = fifo_ref_0;
      ptr_delay_8 < 7 ? (++ptr_delay_8) : (ptr_delay_8 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_7 = 0;
  int16_t fifo_ref_0_delayed_7_buf[7];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_7_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_7_buf distance = 7
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const int16_t let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
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
module_9:
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
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<3> ptr_delay_7 = 0;
  int16_t fifo_ref_0_delayed_7_buf[7];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_7_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_10:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_7_buf distance = 7
    if (!fifo_ld_0.empty())
    {
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const int16_t let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_10
} // Module10Func

void Module11Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_5)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#pragma HLS aggregate bit variable = fifo_ld_4
#pragma HLS aggregate bit variable = fifo_ld_5
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_11:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty())
    {
      int16_t fifo_ref_0;
      int16_t fifo_ref_1;
      int16_t fifo_ref_2;
      int16_t fifo_ref_3;
      int16_t fifo_ref_4;
      int16_t fifo_ref_5;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t((int16_t)((fifo_ref_0 - fifo_ref_1) + ((fifo_ref_2 - fifo_ref_3) * 3) + (fifo_ref_4 - fifo_ref_5))), enabled);
    } // if not empty
  } // for module_11
} // Module11Func

void Module12Func(
  /*output*/ hls::stream<Data<int16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0)
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
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_12
} // Module12Func

void Module13Func(
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
  ap_uint<3> ptr_delay_7 = 0;
  int16_t fifo_ref_0_delayed_7_buf[7];
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
      int16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int16_t fifo_ref_0_delayed_7 = fifo_ref_0_delayed_7_buf[ptr_delay_7];;
      const int16_t let_0 = fifo_ref_0_delayed_7;
      WriteData(fifo_st_0, int16_t(let_0), enabled);
      WriteData(fifo_st_1, int16_t(let_0), enabled);
      WriteData(fifo_st_2, int16_t(let_0), enabled);
      WriteData(fifo_st_3, int16_t(let_0), enabled);
      fifo_ref_0_delayed_7_buf[ptr_delay_7] = fifo_ref_0;
      ptr_delay_7 < 6 ? (++ptr_delay_7) : (ptr_delay_7 = 0);
    } // if not empty
  } // for module_13
} // Module13Func

void Module14Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_5)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#pragma HLS aggregate bit variable = fifo_ld_4
#pragma HLS aggregate bit variable = fifo_ld_5
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_14:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty())
    {
      int16_t fifo_ref_0;
      int16_t fifo_ref_1;
      int16_t fifo_ref_2;
      int16_t fifo_ref_3;
      int16_t fifo_ref_4;
      int16_t fifo_ref_5;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t((uint16_t)((fifo_ref_0 - fifo_ref_1) + ((fifo_ref_2 - fifo_ref_3) * 3) + (fifo_ref_4 - fifo_ref_5))), enabled);
    } // if not empty
  } // for module_14
} // Module14Func

void Module15Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_15:
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
    } // if not empty
  } // for module_15
} // Module15Func

void Module16Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<int16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_16:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty())
    {
      int16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t((uint16_t)(65535 - ((fifo_ref_0 * fifo_ref_0) + (fifo_ref_1 * fifo_ref_1)))), enabled);
    } // if not empty
  } // for module_16
} // Module16Func

void Module17Func(
  /*output*/ hls::stream<Data<ap_uint<64>>>& dram_mag_bank_1_fifo, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = dram_mag_bank_1_fifo
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_17:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      uint16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      uint16_t fifo_ref_2;
      uint16_t fifo_ref_3;
      ap_uint<64> dram_mag_bank_1_buf;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      dram_mag_bank_1_buf(63, 48) = Reinterpret<ap_uint<16>>(fifo_ref_0);
      dram_mag_bank_1_buf(47, 32) = Reinterpret<ap_uint<16>>(fifo_ref_1);
      dram_mag_bank_1_buf(31, 16) = Reinterpret<ap_uint<16>>(fifo_ref_2);
      dram_mag_bank_1_buf(15, 0) = Reinterpret<ap_uint<16>>(fifo_ref_3);
      WriteData(dram_mag_bank_1_fifo, dram_mag_bank_1_buf, enabled);
    } // if not empty
  } // for module_17
} // Module17Func

extern "C" {

void sobel2d_kernel(
  ap_uint<64>* bank_1_mag, 
  ap_uint<64>* bank_0_img, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_mag offset = slave bundle = mag_bank_1
#pragma HLS interface m_axi port = bank_0_img offset = slave bundle = img_bank_0
#pragma HLS interface s_axilite port = bank_1_mag bundle = control
#pragma HLS interface s_axilite port = bank_0_img bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<64>>> bank_0_img_buf("bank_0_img_buf");
#pragma HLS stream variable = bank_0_img_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_img_buf
  hls::stream<Data<ap_uint<64>>> bank_1_mag_buf("bank_1_mag_buf");
#pragma HLS stream variable = bank_1_mag_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_mag_buf

  hls::stream<Data<int16_t>> from_img_bank_0_to_img_offset_2("from_img_bank_0_to_img_offset_2");
#pragma HLS stream variable = from_img_bank_0_to_img_offset_2 depth = 7
#pragma HLS aggregate bit variable = from_img_bank_0_to_img_offset_2
  hls::stream<Data<int16_t>> from_img_bank_0_to_img_offset_0("from_img_bank_0_to_img_offset_0");
#pragma HLS stream variable = from_img_bank_0_to_img_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_img_bank_0_to_img_offset_0
  hls::stream<Data<int16_t>> from_img_bank_0_to_img_offset_3("from_img_bank_0_to_img_offset_3");
#pragma HLS stream variable = from_img_bank_0_to_img_offset_3 depth = 7
#pragma HLS aggregate bit variable = from_img_bank_0_to_img_offset_3
  hls::stream<Data<int16_t>> from_img_bank_0_to_img_offset_1("from_img_bank_0_to_img_offset_1");
#pragma HLS stream variable = from_img_bank_0_to_img_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_img_bank_0_to_img_offset_1
  hls::stream<Data<int16_t>> from_img_offset_2_to_img_offset_34("from_img_offset_2_to_img_offset_34");
#pragma HLS stream variable = from_img_offset_2_to_img_offset_34 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_2_to_img_offset_34
  hls::stream<Data<int16_t>> from_img_offset_2_to_mag_x_pe_3("from_img_offset_2_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_2_to_mag_x_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_2_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_2_to_mag_x_pe_1("from_img_offset_2_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_2_to_mag_x_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_img_offset_2_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_2_to_mag_y_pe_3("from_img_offset_2_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_2_to_mag_y_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_2_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_img_offset_2_to_mag_y_pe_2("from_img_offset_2_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_2_to_mag_y_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_2_to_mag_y_pe_2
  hls::stream<Data<int16_t>> from_img_offset_2_to_mag_y_pe_1("from_img_offset_2_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_2_to_mag_y_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_img_offset_2_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_0_to_img_offset_4("from_img_offset_0_to_img_offset_4");
#pragma HLS stream variable = from_img_offset_0_to_img_offset_4 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_0_to_img_offset_4
  hls::stream<Data<int16_t>> from_img_offset_0_to_mag_x_pe_3("from_img_offset_0_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_0_to_mag_x_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_0_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_0_to_mag_y_pe_3("from_img_offset_0_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_0_to_mag_y_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_0_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_img_offset_3_to_img_offset_35("from_img_offset_3_to_img_offset_35");
#pragma HLS stream variable = from_img_offset_3_to_img_offset_35 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_3_to_img_offset_35
  hls::stream<Data<int16_t>> from_img_offset_3_to_mag_x_pe_2("from_img_offset_3_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_3_to_mag_x_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_3_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_3_to_mag_x_pe_0("from_img_offset_3_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_3_to_mag_x_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_img_offset_3_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_3_to_mag_y_pe_2("from_img_offset_3_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_3_to_mag_y_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_3_to_mag_y_pe_2
  hls::stream<Data<int16_t>> from_img_offset_3_to_mag_y_pe_1("from_img_offset_3_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_3_to_mag_y_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_img_offset_3_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_3_to_mag_y_pe_0("from_img_offset_3_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_3_to_mag_y_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_img_offset_3_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_img_offset_1_to_img_offset_5("from_img_offset_1_to_img_offset_5");
#pragma HLS stream variable = from_img_offset_1_to_img_offset_5 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_1_to_img_offset_5
  hls::stream<Data<int16_t>> from_img_offset_1_to_mag_x_pe_2("from_img_offset_1_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_1_to_mag_x_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_1_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_1_to_mag_y_pe_3("from_img_offset_1_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_1_to_mag_y_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_1_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_img_offset_1_to_mag_y_pe_2("from_img_offset_1_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_1_to_mag_y_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_1_to_mag_y_pe_2
  hls::stream<Data<int16_t>> from_img_offset_34_to_img_offset_66("from_img_offset_34_to_img_offset_66");
#pragma HLS stream variable = from_img_offset_34_to_img_offset_66 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_34_to_img_offset_66
  hls::stream<Data<int16_t>> from_img_offset_34_to_mag_x_pe_3("from_img_offset_34_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_34_to_mag_x_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_34_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_34_to_mag_x_pe_1("from_img_offset_34_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_34_to_mag_x_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_34_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_4_to_img_offset_32("from_img_offset_4_to_img_offset_32");
#pragma HLS stream variable = from_img_offset_4_to_img_offset_32 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_4_to_img_offset_32
  hls::stream<Data<int16_t>> from_img_offset_4_to_mag_x_pe_1("from_img_offset_4_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_4_to_mag_x_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_4_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_4_to_mag_y_pe_1("from_img_offset_4_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_4_to_mag_y_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_4_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_4_to_mag_y_pe_0("from_img_offset_4_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_4_to_mag_y_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_4_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_img_offset_35_to_img_offset_67("from_img_offset_35_to_img_offset_67");
#pragma HLS stream variable = from_img_offset_35_to_img_offset_67 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_35_to_img_offset_67
  hls::stream<Data<int16_t>> from_img_offset_35_to_mag_x_pe_2("from_img_offset_35_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_35_to_mag_x_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_35_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_35_to_mag_x_pe_0("from_img_offset_35_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_35_to_mag_x_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_35_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_5_to_img_offset_33("from_img_offset_5_to_img_offset_33");
#pragma HLS stream variable = from_img_offset_5_to_img_offset_33 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_5_to_img_offset_33
  hls::stream<Data<int16_t>> from_img_offset_5_to_mag_x_pe_0("from_img_offset_5_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_5_to_mag_x_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_5_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_5_to_mag_y_pe_0("from_img_offset_5_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_5_to_mag_y_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_img_offset_5_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_img_offset_66_to_mag_x_pe_3("from_img_offset_66_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_66_to_mag_x_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_66_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_66_to_mag_x_pe_1("from_img_offset_66_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_66_to_mag_x_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_66_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_66_to_mag_y_pe_3("from_img_offset_66_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_66_to_mag_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_66_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_img_offset_66_to_mag_y_pe_2("from_img_offset_66_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_66_to_mag_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_66_to_mag_y_pe_2
  hls::stream<Data<int16_t>> from_img_offset_66_to_mag_y_pe_1("from_img_offset_66_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_66_to_mag_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_66_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_32_to_img_offset_36("from_img_offset_32_to_img_offset_36");
#pragma HLS stream variable = from_img_offset_32_to_img_offset_36 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_32_to_img_offset_36
  hls::stream<Data<int16_t>> from_img_offset_32_to_mag_x_pe_3("from_img_offset_32_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_32_to_mag_x_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_32_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_67_to_mag_x_pe_2("from_img_offset_67_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_67_to_mag_x_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_67_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_67_to_mag_x_pe_0("from_img_offset_67_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_67_to_mag_x_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_67_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_67_to_mag_y_pe_2("from_img_offset_67_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_67_to_mag_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_67_to_mag_y_pe_2
  hls::stream<Data<int16_t>> from_img_offset_67_to_mag_y_pe_1("from_img_offset_67_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_67_to_mag_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_67_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_67_to_mag_y_pe_0("from_img_offset_67_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_67_to_mag_y_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_img_offset_67_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_img_offset_33_to_img_offset_37("from_img_offset_33_to_img_offset_37");
#pragma HLS stream variable = from_img_offset_33_to_img_offset_37 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_33_to_img_offset_37
  hls::stream<Data<int16_t>> from_img_offset_33_to_mag_x_pe_2("from_img_offset_33_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_33_to_mag_x_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_33_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_36_to_img_offset_64("from_img_offset_36_to_img_offset_64");
#pragma HLS stream variable = from_img_offset_36_to_img_offset_64 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_36_to_img_offset_64
  hls::stream<Data<int16_t>> from_img_offset_36_to_mag_x_pe_1("from_img_offset_36_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_36_to_mag_x_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_36_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_37_to_img_offset_65("from_img_offset_37_to_img_offset_65");
#pragma HLS stream variable = from_img_offset_37_to_img_offset_65 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_37_to_img_offset_65
  hls::stream<Data<int16_t>> from_img_offset_37_to_mag_x_pe_0("from_img_offset_37_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_37_to_mag_x_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_img_offset_37_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_64_to_img_offset_68("from_img_offset_64_to_img_offset_68");
#pragma HLS stream variable = from_img_offset_64_to_img_offset_68 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_64_to_img_offset_68
  hls::stream<Data<int16_t>> from_img_offset_64_to_mag_x_pe_3("from_img_offset_64_to_mag_x_pe_3");
#pragma HLS stream variable = from_img_offset_64_to_mag_x_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_64_to_mag_x_pe_3
  hls::stream<Data<int16_t>> from_img_offset_64_to_mag_y_pe_3("from_img_offset_64_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_64_to_mag_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_64_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_mag_x_pe_3_to_mag_x_offset_0("from_mag_x_pe_3_to_mag_x_offset_0");
#pragma HLS stream variable = from_mag_x_pe_3_to_mag_x_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_pe_3_to_mag_x_offset_0
  hls::stream<Data<int16_t>> from_mag_x_offset_0_to_mag_pe_3("from_mag_x_offset_0_to_mag_pe_3");
#pragma HLS stream variable = from_mag_x_offset_0_to_mag_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_offset_0_to_mag_pe_3
  hls::stream<Data<int16_t>> from_img_offset_65_to_img_offset_69("from_img_offset_65_to_img_offset_69");
#pragma HLS stream variable = from_img_offset_65_to_img_offset_69 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_65_to_img_offset_69
  hls::stream<Data<int16_t>> from_img_offset_65_to_mag_x_pe_2("from_img_offset_65_to_mag_x_pe_2");
#pragma HLS stream variable = from_img_offset_65_to_mag_x_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_65_to_mag_x_pe_2
  hls::stream<Data<int16_t>> from_img_offset_65_to_mag_y_pe_3("from_img_offset_65_to_mag_y_pe_3");
#pragma HLS stream variable = from_img_offset_65_to_mag_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_65_to_mag_y_pe_3
  hls::stream<Data<int16_t>> from_img_offset_65_to_mag_y_pe_2("from_img_offset_65_to_mag_y_pe_2");
#pragma HLS stream variable = from_img_offset_65_to_mag_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_65_to_mag_y_pe_2
  hls::stream<Data<uint16_t>> from_mag_y_pe_3_to_mag_y_offset_0("from_mag_y_pe_3_to_mag_y_offset_0");
#pragma HLS stream variable = from_mag_y_pe_3_to_mag_y_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_pe_3_to_mag_y_offset_0
  hls::stream<Data<uint16_t>> from_mag_y_pe_2_to_mag_y_offset_1("from_mag_y_pe_2_to_mag_y_offset_1");
#pragma HLS stream variable = from_mag_y_pe_2_to_mag_y_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_pe_2_to_mag_y_offset_1
  hls::stream<Data<int16_t>> from_mag_x_pe_2_to_mag_x_offset_1("from_mag_x_pe_2_to_mag_x_offset_1");
#pragma HLS stream variable = from_mag_x_pe_2_to_mag_x_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_pe_2_to_mag_x_offset_1
  hls::stream<Data<uint16_t>> from_mag_y_offset_0_to_mag_pe_3("from_mag_y_offset_0_to_mag_pe_3");
#pragma HLS stream variable = from_mag_y_offset_0_to_mag_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_offset_0_to_mag_pe_3
  hls::stream<Data<uint16_t>> from_mag_y_offset_1_to_mag_pe_2("from_mag_y_offset_1_to_mag_pe_2");
#pragma HLS stream variable = from_mag_y_offset_1_to_mag_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_offset_1_to_mag_pe_2
  hls::stream<Data<int16_t>> from_mag_x_offset_1_to_mag_pe_2("from_mag_x_offset_1_to_mag_pe_2");
#pragma HLS stream variable = from_mag_x_offset_1_to_mag_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_offset_1_to_mag_pe_2
  hls::stream<Data<uint16_t>> from_mag_pe_3_to_mag_bank_1("from_mag_pe_3_to_mag_bank_1");
#pragma HLS stream variable = from_mag_pe_3_to_mag_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_mag_pe_3_to_mag_bank_1
  hls::stream<Data<uint16_t>> from_mag_pe_2_to_mag_bank_1("from_mag_pe_2_to_mag_bank_1");
#pragma HLS stream variable = from_mag_pe_2_to_mag_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_mag_pe_2_to_mag_bank_1
  hls::stream<Data<int16_t>> from_img_offset_68_to_mag_x_pe_1("from_img_offset_68_to_mag_x_pe_1");
#pragma HLS stream variable = from_img_offset_68_to_mag_x_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_68_to_mag_x_pe_1
  hls::stream<Data<int16_t>> from_img_offset_68_to_mag_y_pe_1("from_img_offset_68_to_mag_y_pe_1");
#pragma HLS stream variable = from_img_offset_68_to_mag_y_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_68_to_mag_y_pe_1
  hls::stream<Data<int16_t>> from_img_offset_68_to_mag_y_pe_0("from_img_offset_68_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_68_to_mag_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_68_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_mag_x_pe_1_to_mag_x_offset_2("from_mag_x_pe_1_to_mag_x_offset_2");
#pragma HLS stream variable = from_mag_x_pe_1_to_mag_x_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_pe_1_to_mag_x_offset_2
  hls::stream<Data<uint16_t>> from_mag_y_pe_1_to_mag_y_offset_2("from_mag_y_pe_1_to_mag_y_offset_2");
#pragma HLS stream variable = from_mag_y_pe_1_to_mag_y_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_pe_1_to_mag_y_offset_2
  hls::stream<Data<int16_t>> from_mag_x_offset_2_to_mag_pe_1("from_mag_x_offset_2_to_mag_pe_1");
#pragma HLS stream variable = from_mag_x_offset_2_to_mag_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_offset_2_to_mag_pe_1
  hls::stream<Data<uint16_t>> from_mag_y_offset_2_to_mag_pe_1("from_mag_y_offset_2_to_mag_pe_1");
#pragma HLS stream variable = from_mag_y_offset_2_to_mag_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_offset_2_to_mag_pe_1
  hls::stream<Data<uint16_t>> from_mag_pe_1_to_mag_bank_1("from_mag_pe_1_to_mag_bank_1");
#pragma HLS stream variable = from_mag_pe_1_to_mag_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_pe_1_to_mag_bank_1
  hls::stream<Data<int16_t>> from_img_offset_69_to_mag_x_pe_0("from_img_offset_69_to_mag_x_pe_0");
#pragma HLS stream variable = from_img_offset_69_to_mag_x_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_69_to_mag_x_pe_0
  hls::stream<Data<int16_t>> from_img_offset_69_to_mag_y_pe_0("from_img_offset_69_to_mag_y_pe_0");
#pragma HLS stream variable = from_img_offset_69_to_mag_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_img_offset_69_to_mag_y_pe_0
  hls::stream<Data<int16_t>> from_mag_x_pe_0_to_mag_x_offset_3("from_mag_x_pe_0_to_mag_x_offset_3");
#pragma HLS stream variable = from_mag_x_pe_0_to_mag_x_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_pe_0_to_mag_x_offset_3
  hls::stream<Data<uint16_t>> from_mag_y_pe_0_to_mag_y_offset_3("from_mag_y_pe_0_to_mag_y_offset_3");
#pragma HLS stream variable = from_mag_y_pe_0_to_mag_y_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_pe_0_to_mag_y_offset_3
  hls::stream<Data<int16_t>> from_mag_x_offset_3_to_mag_pe_0("from_mag_x_offset_3_to_mag_pe_0");
#pragma HLS stream variable = from_mag_x_offset_3_to_mag_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_mag_x_offset_3_to_mag_pe_0
  hls::stream<Data<uint16_t>> from_mag_y_offset_3_to_mag_pe_0("from_mag_y_offset_3_to_mag_pe_0");
#pragma HLS stream variable = from_mag_y_offset_3_to_mag_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_mag_y_offset_3_to_mag_pe_0
  hls::stream<Data<uint16_t>> from_mag_pe_0_to_mag_bank_1("from_mag_pe_0_to_mag_bank_1");
#pragma HLS stream variable = from_mag_pe_0_to_mag_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_mag_pe_0_to_mag_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_img_buf, bank_0_img, coalesced_data_num);
  Module0Func(
      /*output*/ from_img_bank_0_to_img_offset_2, 
      /*output*/ from_img_bank_0_to_img_offset_0, 
      /*output*/ from_img_bank_0_to_img_offset_3, 
      /*output*/ from_img_bank_0_to_img_offset_1, 
      /* input*/ bank_0_img_buf);
  Module1Func(
      /*output*/ from_img_offset_2_to_img_offset_34, 
      /*output*/ from_img_offset_2_to_mag_x_pe_3, 
      /*output*/ from_img_offset_2_to_mag_x_pe_1, 
      /*output*/ from_img_offset_2_to_mag_y_pe_3, 
      /*output*/ from_img_offset_2_to_mag_y_pe_2, 
      /*output*/ from_img_offset_2_to_mag_y_pe_1, 
      /* input*/ from_img_bank_0_to_img_offset_2);
  Module2Func(
      /*output*/ from_img_offset_0_to_img_offset_4, 
      /*output*/ from_img_offset_0_to_mag_x_pe_3, 
      /*output*/ from_img_offset_0_to_mag_y_pe_3, 
      /* input*/ from_img_bank_0_to_img_offset_0);
  Module1Func(
      /*output*/ from_img_offset_3_to_img_offset_35, 
      /*output*/ from_img_offset_3_to_mag_x_pe_2, 
      /*output*/ from_img_offset_3_to_mag_x_pe_0, 
      /*output*/ from_img_offset_3_to_mag_y_pe_2, 
      /*output*/ from_img_offset_3_to_mag_y_pe_1, 
      /*output*/ from_img_offset_3_to_mag_y_pe_0, 
      /* input*/ from_img_bank_0_to_img_offset_3);
  Module3Func(
      /*output*/ from_img_offset_1_to_img_offset_5, 
      /*output*/ from_img_offset_1_to_mag_x_pe_2, 
      /*output*/ from_img_offset_1_to_mag_y_pe_3, 
      /*output*/ from_img_offset_1_to_mag_y_pe_2, 
      /* input*/ from_img_bank_0_to_img_offset_1);
  Module4Func(
      /*output*/ from_img_offset_34_to_img_offset_66, 
      /*output*/ from_img_offset_34_to_mag_x_pe_3, 
      /*output*/ from_img_offset_34_to_mag_x_pe_1, 
      /* input*/ from_img_offset_2_to_img_offset_34);
  Module5Func(
      /*output*/ from_img_offset_4_to_img_offset_32, 
      /*output*/ from_img_offset_4_to_mag_x_pe_1, 
      /*output*/ from_img_offset_4_to_mag_y_pe_1, 
      /*output*/ from_img_offset_4_to_mag_y_pe_0, 
      /* input*/ from_img_offset_0_to_img_offset_4);
  Module4Func(
      /*output*/ from_img_offset_35_to_img_offset_67, 
      /*output*/ from_img_offset_35_to_mag_x_pe_2, 
      /*output*/ from_img_offset_35_to_mag_x_pe_0, 
      /* input*/ from_img_offset_3_to_img_offset_35);
  Module6Func(
      /*output*/ from_img_offset_5_to_img_offset_33, 
      /*output*/ from_img_offset_5_to_mag_x_pe_0, 
      /*output*/ from_img_offset_5_to_mag_y_pe_0, 
      /* input*/ from_img_offset_1_to_img_offset_5);
  Module7Func(
      /*output*/ from_img_offset_66_to_mag_x_pe_3, 
      /*output*/ from_img_offset_66_to_mag_x_pe_1, 
      /*output*/ from_img_offset_66_to_mag_y_pe_3, 
      /*output*/ from_img_offset_66_to_mag_y_pe_2, 
      /*output*/ from_img_offset_66_to_mag_y_pe_1, 
      /* input*/ from_img_offset_34_to_img_offset_66);
  Module8Func(
      /*output*/ from_img_offset_32_to_img_offset_36, 
      /*output*/ from_img_offset_32_to_mag_x_pe_3, 
      /* input*/ from_img_offset_4_to_img_offset_32);
  Module7Func(
      /*output*/ from_img_offset_67_to_mag_x_pe_2, 
      /*output*/ from_img_offset_67_to_mag_x_pe_0, 
      /*output*/ from_img_offset_67_to_mag_y_pe_2, 
      /*output*/ from_img_offset_67_to_mag_y_pe_1, 
      /*output*/ from_img_offset_67_to_mag_y_pe_0, 
      /* input*/ from_img_offset_35_to_img_offset_67);
  Module8Func(
      /*output*/ from_img_offset_33_to_img_offset_37, 
      /*output*/ from_img_offset_33_to_mag_x_pe_2, 
      /* input*/ from_img_offset_5_to_img_offset_33);
  Module9Func(
      /*output*/ from_img_offset_36_to_img_offset_64, 
      /*output*/ from_img_offset_36_to_mag_x_pe_1, 
      /* input*/ from_img_offset_32_to_img_offset_36);
  Module9Func(
      /*output*/ from_img_offset_37_to_img_offset_65, 
      /*output*/ from_img_offset_37_to_mag_x_pe_0, 
      /* input*/ from_img_offset_33_to_img_offset_37);
  Module10Func(
      /*output*/ from_img_offset_64_to_img_offset_68, 
      /*output*/ from_img_offset_64_to_mag_x_pe_3, 
      /*output*/ from_img_offset_64_to_mag_y_pe_3, 
      /* input*/ from_img_offset_36_to_img_offset_64);
  Module11Func(
      /*output*/ from_mag_x_pe_3_to_mag_x_offset_0, 
      /* input*/ from_img_offset_64_to_mag_x_pe_3, 
      /* input*/ from_img_offset_66_to_mag_x_pe_3, 
      /* input*/ from_img_offset_32_to_mag_x_pe_3, 
      /* input*/ from_img_offset_34_to_mag_x_pe_3, 
      /* input*/ from_img_offset_0_to_mag_x_pe_3, 
      /* input*/ from_img_offset_2_to_mag_x_pe_3);
  Module12Func(
      /*output*/ from_mag_x_offset_0_to_mag_pe_3, 
      /* input*/ from_mag_x_pe_3_to_mag_x_offset_0);
  Module13Func(
      /*output*/ from_img_offset_65_to_img_offset_69, 
      /*output*/ from_img_offset_65_to_mag_x_pe_2, 
      /*output*/ from_img_offset_65_to_mag_y_pe_3, 
      /*output*/ from_img_offset_65_to_mag_y_pe_2, 
      /* input*/ from_img_offset_37_to_img_offset_65);
  Module14Func(
      /*output*/ from_mag_y_pe_3_to_mag_y_offset_0, 
      /* input*/ from_img_offset_2_to_mag_y_pe_3, 
      /* input*/ from_img_offset_66_to_mag_y_pe_3, 
      /* input*/ from_img_offset_1_to_mag_y_pe_3, 
      /* input*/ from_img_offset_65_to_mag_y_pe_3, 
      /* input*/ from_img_offset_0_to_mag_y_pe_3, 
      /* input*/ from_img_offset_64_to_mag_y_pe_3);
  Module14Func(
      /*output*/ from_mag_y_pe_2_to_mag_y_offset_1, 
      /* input*/ from_img_offset_3_to_mag_y_pe_2, 
      /* input*/ from_img_offset_67_to_mag_y_pe_2, 
      /* input*/ from_img_offset_2_to_mag_y_pe_2, 
      /* input*/ from_img_offset_66_to_mag_y_pe_2, 
      /* input*/ from_img_offset_1_to_mag_y_pe_2, 
      /* input*/ from_img_offset_65_to_mag_y_pe_2);
  Module11Func(
      /*output*/ from_mag_x_pe_2_to_mag_x_offset_1, 
      /* input*/ from_img_offset_65_to_mag_x_pe_2, 
      /* input*/ from_img_offset_67_to_mag_x_pe_2, 
      /* input*/ from_img_offset_33_to_mag_x_pe_2, 
      /* input*/ from_img_offset_35_to_mag_x_pe_2, 
      /* input*/ from_img_offset_1_to_mag_x_pe_2, 
      /* input*/ from_img_offset_3_to_mag_x_pe_2);
  Module15Func(
      /*output*/ from_mag_y_offset_0_to_mag_pe_3, 
      /* input*/ from_mag_y_pe_3_to_mag_y_offset_0);
  Module15Func(
      /*output*/ from_mag_y_offset_1_to_mag_pe_2, 
      /* input*/ from_mag_y_pe_2_to_mag_y_offset_1);
  Module12Func(
      /*output*/ from_mag_x_offset_1_to_mag_pe_2, 
      /* input*/ from_mag_x_pe_2_to_mag_x_offset_1);
  Module16Func(
      /*output*/ from_mag_pe_3_to_mag_bank_1, 
      /* input*/ from_mag_x_offset_0_to_mag_pe_3, 
      /* input*/ from_mag_y_offset_0_to_mag_pe_3);
  Module16Func(
      /*output*/ from_mag_pe_2_to_mag_bank_1, 
      /* input*/ from_mag_x_offset_1_to_mag_pe_2, 
      /* input*/ from_mag_y_offset_1_to_mag_pe_2);
  Module6Func(
      /*output*/ from_img_offset_68_to_mag_x_pe_1, 
      /*output*/ from_img_offset_68_to_mag_y_pe_1, 
      /*output*/ from_img_offset_68_to_mag_y_pe_0, 
      /* input*/ from_img_offset_64_to_img_offset_68);
  Module11Func(
      /*output*/ from_mag_x_pe_1_to_mag_x_offset_2, 
      /* input*/ from_img_offset_66_to_mag_x_pe_1, 
      /* input*/ from_img_offset_68_to_mag_x_pe_1, 
      /* input*/ from_img_offset_34_to_mag_x_pe_1, 
      /* input*/ from_img_offset_36_to_mag_x_pe_1, 
      /* input*/ from_img_offset_2_to_mag_x_pe_1, 
      /* input*/ from_img_offset_4_to_mag_x_pe_1);
  Module14Func(
      /*output*/ from_mag_y_pe_1_to_mag_y_offset_2, 
      /* input*/ from_img_offset_4_to_mag_y_pe_1, 
      /* input*/ from_img_offset_68_to_mag_y_pe_1, 
      /* input*/ from_img_offset_3_to_mag_y_pe_1, 
      /* input*/ from_img_offset_67_to_mag_y_pe_1, 
      /* input*/ from_img_offset_2_to_mag_y_pe_1, 
      /* input*/ from_img_offset_66_to_mag_y_pe_1);
  Module12Func(
      /*output*/ from_mag_x_offset_2_to_mag_pe_1, 
      /* input*/ from_mag_x_pe_1_to_mag_x_offset_2);
  Module15Func(
      /*output*/ from_mag_y_offset_2_to_mag_pe_1, 
      /* input*/ from_mag_y_pe_1_to_mag_y_offset_2);
  Module16Func(
      /*output*/ from_mag_pe_1_to_mag_bank_1, 
      /* input*/ from_mag_x_offset_2_to_mag_pe_1, 
      /* input*/ from_mag_y_offset_2_to_mag_pe_1);
  Module9Func(
      /*output*/ from_img_offset_69_to_mag_x_pe_0, 
      /*output*/ from_img_offset_69_to_mag_y_pe_0, 
      /* input*/ from_img_offset_65_to_img_offset_69);
  Module11Func(
      /*output*/ from_mag_x_pe_0_to_mag_x_offset_3, 
      /* input*/ from_img_offset_67_to_mag_x_pe_0, 
      /* input*/ from_img_offset_69_to_mag_x_pe_0, 
      /* input*/ from_img_offset_35_to_mag_x_pe_0, 
      /* input*/ from_img_offset_37_to_mag_x_pe_0, 
      /* input*/ from_img_offset_3_to_mag_x_pe_0, 
      /* input*/ from_img_offset_5_to_mag_x_pe_0);
  Module14Func(
      /*output*/ from_mag_y_pe_0_to_mag_y_offset_3, 
      /* input*/ from_img_offset_5_to_mag_y_pe_0, 
      /* input*/ from_img_offset_69_to_mag_y_pe_0, 
      /* input*/ from_img_offset_4_to_mag_y_pe_0, 
      /* input*/ from_img_offset_68_to_mag_y_pe_0, 
      /* input*/ from_img_offset_3_to_mag_y_pe_0, 
      /* input*/ from_img_offset_67_to_mag_y_pe_0);
  Module12Func(
      /*output*/ from_mag_x_offset_3_to_mag_pe_0, 
      /* input*/ from_mag_x_pe_0_to_mag_x_offset_3);
  Module15Func(
      /*output*/ from_mag_y_offset_3_to_mag_pe_0, 
      /* input*/ from_mag_y_pe_0_to_mag_y_offset_3);
  Module16Func(
      /*output*/ from_mag_pe_0_to_mag_bank_1, 
      /* input*/ from_mag_x_offset_3_to_mag_pe_0, 
      /* input*/ from_mag_y_offset_3_to_mag_pe_0);
  Module17Func(
      /*output*/ bank_1_mag_buf, 
      /* input*/ from_mag_pe_3_to_mag_bank_1, 
      /* input*/ from_mag_pe_2_to_mag_bank_1, 
      /* input*/ from_mag_pe_1_to_mag_bank_1, 
      /* input*/ from_mag_pe_0_to_mag_bank_1);
  BurstWrite(bank_1_mag, bank_1_mag_buf, coalesced_data_num);
}

}  // extern "C"
