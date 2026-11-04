// this file can be generated from the following SODA DSL
/*
kernel: jacobi3d
burst width: 64
iterate: 2
unroll factor: 2
input dram 0 float: t1(32, 32, *)
output dram 1 float: t0(0, 0, 0) = (t1(0, 0, 0) + t1(1, 0, 0) + t1(-1, 0, 0) + t1(0, 1, 0) + t1(0, -1, 0) + t1(0, 0, 1) + t1(0, 0, -1)) * 0.142857142f
border: ignore
cluster: none
*/

// stencil window size: (5, 5, 5)
// stencil distace: 4162
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
  /* input*/ hls::stream<Data<ap_uint<64>>>& dram_t1_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_t1_bank_0_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_0:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!dram_t1_bank_0_fifo.empty())
    {
      ap_uint<64> dram_t1_bank_0_buf;
      const bool dram_t1_bank_0_buf_enable = ReadData(dram_t1_bank_0_buf, dram_t1_bank_0_fifo);
      const bool enabled = dram_t1_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_t1_bank_0_buf(63, 32))), enabled);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_t1_bank_0_buf(31, 0))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
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
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
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
  ap_uint<9> ptr_delay_496 = 0;
  float fifo_ref_0_delayed_496_buf[496];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_496_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_2:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_496_buf distance = 496
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_496 = fifo_ref_0_delayed_496_buf[ptr_delay_496];;
      const float let_0 = fifo_ref_0_delayed_496;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_496_buf[ptr_delay_496] = fifo_ref_0;
      ptr_delay_496 < 495 ? (++ptr_delay_496) : (ptr_delay_496 = 0);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<4> ptr_delay_16 = 0;
  float fifo_ref_0_delayed_16_buf[16];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_16_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_3:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_16_buf distance = 16
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_16 = fifo_ref_0_delayed_16_buf[ptr_delay_16];;
      const float let_0 = fifo_ref_0_delayed_16;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_16_buf[ptr_delay_16] = fifo_ref_0;
      ptr_delay_16 < 15 ? (++ptr_delay_16) : (ptr_delay_16 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<4> ptr_delay_15 = 0;
  float fifo_ref_0_delayed_15_buf[15];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_15_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_15_buf distance = 15
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_15 = fifo_ref_0_delayed_15_buf[ptr_delay_15];;
      const float let_0 = fifo_ref_0_delayed_15;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_15_buf[ptr_delay_15] = fifo_ref_0;
      ptr_delay_15 < 14 ? (++ptr_delay_15) : (ptr_delay_15 = 0);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
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
module_5:
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
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
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
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
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
  ap_uint<4> ptr_delay_16 = 0;
  float fifo_ref_0_delayed_16_buf[16];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_16_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_7:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_16_buf distance = 16
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_16 = fifo_ref_0_delayed_16_buf[ptr_delay_16];;
      const float let_0 = fifo_ref_0_delayed_16;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_16_buf[ptr_delay_16] = fifo_ref_0;
      ptr_delay_16 < 15 ? (++ptr_delay_16) : (ptr_delay_16 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<9> ptr_delay_496 = 0;
  float fifo_ref_0_delayed_496_buf[496];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_496_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_496_buf distance = 496
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_496 = fifo_ref_0_delayed_496_buf[ptr_delay_496];;
      const float let_0 = fifo_ref_0_delayed_496;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_496_buf[ptr_delay_496] = fifo_ref_0;
      ptr_delay_496 < 495 ? (++ptr_delay_496) : (ptr_delay_496 = 0);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_6)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#pragma HLS aggregate bit variable = fifo_ld_4
#pragma HLS aggregate bit variable = fifo_ld_5
#pragma HLS aggregate bit variable = fifo_ld_6
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_9:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      float fifo_ref_5;
      float fifo_ref_6;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool fifo_ref_6_enable = ReadData(fifo_ref_6, fifo_ld_6);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)((fifo_ref_0 + fifo_ref_1 + fifo_ref_2 + fifo_ref_3 + fifo_ref_4 + fifo_ref_5 + fifo_ref_6) * 0.142857142f)), enabled);
    } // if not empty
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<ap_uint<64>>>& dram_t0_bank_1_fifo, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1)
{
#pragma HLS aggregate bit variable = dram_t0_bank_1_fifo
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
      ap_uint<64> dram_t0_bank_1_buf;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable;
      enable = enabled;
      dram_t0_bank_1_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_t0_bank_1_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      WriteData(dram_t0_bank_1_fifo, dram_t0_bank_1_buf, enabled);
    } // if not empty
  } // for module_10
} // Module10Func

extern "C" {

void jacobi3d_kernel(
  ap_uint<64>* bank_1_t0, 
  ap_uint<64>* bank_0_t1, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_t0 offset = slave bundle = t0_bank_1
#pragma HLS interface m_axi port = bank_0_t1 offset = slave bundle = t1_bank_0
#pragma HLS interface s_axilite port = bank_1_t0 bundle = control
#pragma HLS interface s_axilite port = bank_0_t1 bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<64>>> bank_0_t1_buf("bank_0_t1_buf");
#pragma HLS stream variable = bank_0_t1_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_t1_buf
  hls::stream<Data<ap_uint<64>>> bank_1_t0_buf("bank_1_t0_buf");
#pragma HLS stream variable = bank_1_t0_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_t0_buf

  hls::stream<Data<float>> from_t1_bank_0_to_t1_offset_0("from_t1_bank_0_to_t1_offset_0");
#pragma HLS stream variable = from_t1_bank_0_to_t1_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_t1_bank_0_to_t1_offset_0
  hls::stream<Data<float>> from_t1_bank_0_to_t1_offset_1("from_t1_bank_0_to_t1_offset_1");
#pragma HLS stream variable = from_t1_bank_0_to_t1_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_t1_bank_0_to_t1_offset_1
  hls::stream<Data<float>> from_t1_offset_0_to_t1_offset_992("from_t1_offset_0_to_t1_offset_992");
#pragma HLS stream variable = from_t1_offset_0_to_t1_offset_992 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_0_to_t1_offset_992
  hls::stream<Data<float>> from_t1_offset_0_to_t1_iter1_pe_1("from_t1_offset_0_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_0_to_t1_iter1_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_t1_offset_0_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_1_to_t1_offset_993("from_t1_offset_1_to_t1_offset_993");
#pragma HLS stream variable = from_t1_offset_1_to_t1_offset_993 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1_to_t1_offset_993
  hls::stream<Data<float>> from_t1_offset_1_to_t1_iter1_pe_0("from_t1_offset_1_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_1_to_t1_iter1_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_t1_offset_1_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_992_to_t1_offset_1024("from_t1_offset_992_to_t1_offset_1024");
#pragma HLS stream variable = from_t1_offset_992_to_t1_offset_1024 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_992_to_t1_offset_1024
  hls::stream<Data<float>> from_t1_offset_992_to_t1_iter1_pe_1("from_t1_offset_992_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_992_to_t1_iter1_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_t1_offset_992_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_993_to_t1_offset_1023("from_t1_offset_993_to_t1_offset_1023");
#pragma HLS stream variable = from_t1_offset_993_to_t1_offset_1023 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_993_to_t1_offset_1023
  hls::stream<Data<float>> from_t1_offset_993_to_t1_iter1_pe_0("from_t1_offset_993_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_993_to_t1_iter1_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_t1_offset_993_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_1024_to_t1_offset_1026("from_t1_offset_1024_to_t1_offset_1026");
#pragma HLS stream variable = from_t1_offset_1024_to_t1_offset_1026 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1024_to_t1_offset_1026
  hls::stream<Data<float>> from_t1_offset_1024_to_t1_iter1_pe_1("from_t1_offset_1024_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_1024_to_t1_iter1_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_t1_offset_1024_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_1024_to_t1_iter1_pe_0("from_t1_offset_1024_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_1024_to_t1_iter1_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_t1_offset_1024_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_1023_to_t1_offset_1025("from_t1_offset_1023_to_t1_offset_1025");
#pragma HLS stream variable = from_t1_offset_1023_to_t1_offset_1025 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1023_to_t1_offset_1025
  hls::stream<Data<float>> from_t1_offset_1023_to_t1_iter1_pe_1("from_t1_offset_1023_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_1023_to_t1_iter1_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_t1_offset_1023_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_1026_to_t1_offset_1056("from_t1_offset_1026_to_t1_offset_1056");
#pragma HLS stream variable = from_t1_offset_1026_to_t1_offset_1056 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1026_to_t1_offset_1056
  hls::stream<Data<float>> from_t1_offset_1026_to_t1_iter1_pe_0("from_t1_offset_1026_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_1026_to_t1_iter1_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_t1_offset_1026_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_1025_to_t1_offset_1057("from_t1_offset_1025_to_t1_offset_1057");
#pragma HLS stream variable = from_t1_offset_1025_to_t1_offset_1057 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1025_to_t1_offset_1057
  hls::stream<Data<float>> from_t1_offset_1025_to_t1_iter1_pe_1("from_t1_offset_1025_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_1025_to_t1_iter1_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_t1_offset_1025_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_1025_to_t1_iter1_pe_0("from_t1_offset_1025_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_1025_to_t1_iter1_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_t1_offset_1025_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_1056_to_t1_offset_2048("from_t1_offset_1056_to_t1_offset_2048");
#pragma HLS stream variable = from_t1_offset_1056_to_t1_offset_2048 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1056_to_t1_offset_2048
  hls::stream<Data<float>> from_t1_offset_1056_to_t1_iter1_pe_1("from_t1_offset_1056_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_1056_to_t1_iter1_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_t1_offset_1056_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_offset_1057_to_t1_offset_2049("from_t1_offset_1057_to_t1_offset_2049");
#pragma HLS stream variable = from_t1_offset_1057_to_t1_offset_2049 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_1057_to_t1_offset_2049
  hls::stream<Data<float>> from_t1_offset_1057_to_t1_iter1_pe_0("from_t1_offset_1057_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_1057_to_t1_iter1_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_t1_offset_1057_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_offset_2048_to_t1_iter1_pe_1("from_t1_offset_2048_to_t1_iter1_pe_1");
#pragma HLS stream variable = from_t1_offset_2048_to_t1_iter1_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_2048_to_t1_iter1_pe_1
  hls::stream<Data<float>> from_t1_iter1_pe_1_to_t1_iter1_offset_0("from_t1_iter1_pe_1_to_t1_iter1_offset_0");
#pragma HLS stream variable = from_t1_iter1_pe_1_to_t1_iter1_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_pe_1_to_t1_iter1_offset_0
  hls::stream<Data<float>> from_t1_iter1_offset_0_to_t1_iter1_offset_992("from_t1_iter1_offset_0_to_t1_iter1_offset_992");
#pragma HLS stream variable = from_t1_iter1_offset_0_to_t1_iter1_offset_992 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_0_to_t1_iter1_offset_992
  hls::stream<Data<float>> from_t1_iter1_offset_0_to_t0_pe_1("from_t1_iter1_offset_0_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_0_to_t0_pe_1 depth = 13
#pragma HLS aggregate bit variable = from_t1_iter1_offset_0_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_992_to_t1_iter1_offset_1024("from_t1_iter1_offset_992_to_t1_iter1_offset_1024");
#pragma HLS stream variable = from_t1_iter1_offset_992_to_t1_iter1_offset_1024 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_992_to_t1_iter1_offset_1024
  hls::stream<Data<float>> from_t1_iter1_offset_992_to_t0_pe_1("from_t1_iter1_offset_992_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_992_to_t0_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_t1_iter1_offset_992_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_1024_to_t1_iter1_offset_1026("from_t1_iter1_offset_1024_to_t1_iter1_offset_1026");
#pragma HLS stream variable = from_t1_iter1_offset_1024_to_t1_iter1_offset_1026 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1024_to_t1_iter1_offset_1026
  hls::stream<Data<float>> from_t1_iter1_offset_1024_to_t0_pe_1("from_t1_iter1_offset_1024_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_1024_to_t0_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1024_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_1024_to_t0_pe_0("from_t1_iter1_offset_1024_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_1024_to_t0_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1024_to_t0_pe_0
  hls::stream<Data<float>> from_t1_iter1_offset_1026_to_t1_iter1_offset_1056("from_t1_iter1_offset_1026_to_t1_iter1_offset_1056");
#pragma HLS stream variable = from_t1_iter1_offset_1026_to_t1_iter1_offset_1056 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1026_to_t1_iter1_offset_1056
  hls::stream<Data<float>> from_t1_iter1_offset_1026_to_t0_pe_0("from_t1_iter1_offset_1026_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_1026_to_t0_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1026_to_t0_pe_0
  hls::stream<Data<float>> from_t1_offset_2049_to_t1_iter1_pe_0("from_t1_offset_2049_to_t1_iter1_pe_0");
#pragma HLS stream variable = from_t1_offset_2049_to_t1_iter1_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_t1_offset_2049_to_t1_iter1_pe_0
  hls::stream<Data<float>> from_t1_iter1_pe_0_to_t1_iter1_offset_1("from_t1_iter1_pe_0_to_t1_iter1_offset_1");
#pragma HLS stream variable = from_t1_iter1_pe_0_to_t1_iter1_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_pe_0_to_t1_iter1_offset_1
  hls::stream<Data<float>> from_t1_iter1_offset_1_to_t1_iter1_offset_993("from_t1_iter1_offset_1_to_t1_iter1_offset_993");
#pragma HLS stream variable = from_t1_iter1_offset_1_to_t1_iter1_offset_993 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1_to_t1_iter1_offset_993
  hls::stream<Data<float>> from_t1_iter1_offset_1_to_t0_pe_0("from_t1_iter1_offset_1_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_1_to_t0_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1_to_t0_pe_0
  hls::stream<Data<float>> from_t1_iter1_offset_993_to_t1_iter1_offset_1023("from_t1_iter1_offset_993_to_t1_iter1_offset_1023");
#pragma HLS stream variable = from_t1_iter1_offset_993_to_t1_iter1_offset_1023 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_993_to_t1_iter1_offset_1023
  hls::stream<Data<float>> from_t1_iter1_offset_993_to_t0_pe_0("from_t1_iter1_offset_993_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_993_to_t0_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_t1_iter1_offset_993_to_t0_pe_0
  hls::stream<Data<float>> from_t1_iter1_offset_1023_to_t1_iter1_offset_1025("from_t1_iter1_offset_1023_to_t1_iter1_offset_1025");
#pragma HLS stream variable = from_t1_iter1_offset_1023_to_t1_iter1_offset_1025 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1023_to_t1_iter1_offset_1025
  hls::stream<Data<float>> from_t1_iter1_offset_1023_to_t0_pe_1("from_t1_iter1_offset_1023_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_1023_to_t0_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1023_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_1025_to_t1_iter1_offset_1057("from_t1_iter1_offset_1025_to_t1_iter1_offset_1057");
#pragma HLS stream variable = from_t1_iter1_offset_1025_to_t1_iter1_offset_1057 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1025_to_t1_iter1_offset_1057
  hls::stream<Data<float>> from_t1_iter1_offset_1025_to_t0_pe_1("from_t1_iter1_offset_1025_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_1025_to_t0_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1025_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_1025_to_t0_pe_0("from_t1_iter1_offset_1025_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_1025_to_t0_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1025_to_t0_pe_0
  hls::stream<Data<float>> from_t1_iter1_offset_1056_to_t1_iter1_offset_2048("from_t1_iter1_offset_1056_to_t1_iter1_offset_2048");
#pragma HLS stream variable = from_t1_iter1_offset_1056_to_t1_iter1_offset_2048 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1056_to_t1_iter1_offset_2048
  hls::stream<Data<float>> from_t1_iter1_offset_1056_to_t0_pe_1("from_t1_iter1_offset_1056_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_1056_to_t0_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1056_to_t0_pe_1
  hls::stream<Data<float>> from_t1_iter1_offset_1057_to_t1_iter1_offset_2049("from_t1_iter1_offset_1057_to_t1_iter1_offset_2049");
#pragma HLS stream variable = from_t1_iter1_offset_1057_to_t1_iter1_offset_2049 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1057_to_t1_iter1_offset_2049
  hls::stream<Data<float>> from_t1_iter1_offset_1057_to_t0_pe_0("from_t1_iter1_offset_1057_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_1057_to_t0_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_t1_iter1_offset_1057_to_t0_pe_0
  hls::stream<Data<float>> from_t1_iter1_offset_2048_to_t0_pe_1("from_t1_iter1_offset_2048_to_t0_pe_1");
#pragma HLS stream variable = from_t1_iter1_offset_2048_to_t0_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_2048_to_t0_pe_1
  hls::stream<Data<float>> from_t0_pe_1_to_t0_bank_1("from_t0_pe_1_to_t0_bank_1");
#pragma HLS stream variable = from_t0_pe_1_to_t0_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_t0_pe_1_to_t0_bank_1
  hls::stream<Data<float>> from_t1_iter1_offset_2049_to_t0_pe_0("from_t1_iter1_offset_2049_to_t0_pe_0");
#pragma HLS stream variable = from_t1_iter1_offset_2049_to_t0_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_t1_iter1_offset_2049_to_t0_pe_0
  hls::stream<Data<float>> from_t0_pe_0_to_t0_bank_1("from_t0_pe_0_to_t0_bank_1");
#pragma HLS stream variable = from_t0_pe_0_to_t0_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_t0_pe_0_to_t0_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_t1_buf, bank_0_t1, coalesced_data_num);
  Module0Func(
      /*output*/ from_t1_bank_0_to_t1_offset_0, 
      /*output*/ from_t1_bank_0_to_t1_offset_1, 
      /* input*/ bank_0_t1_buf);
  Module1Func(
      /*output*/ from_t1_offset_0_to_t1_offset_992, 
      /*output*/ from_t1_offset_0_to_t1_iter1_pe_1, 
      /* input*/ from_t1_bank_0_to_t1_offset_0);
  Module1Func(
      /*output*/ from_t1_offset_1_to_t1_offset_993, 
      /*output*/ from_t1_offset_1_to_t1_iter1_pe_0, 
      /* input*/ from_t1_bank_0_to_t1_offset_1);
  Module2Func(
      /*output*/ from_t1_offset_992_to_t1_offset_1024, 
      /*output*/ from_t1_offset_992_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_0_to_t1_offset_992);
  Module2Func(
      /*output*/ from_t1_offset_993_to_t1_offset_1023, 
      /*output*/ from_t1_offset_993_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1_to_t1_offset_993);
  Module3Func(
      /*output*/ from_t1_offset_1024_to_t1_offset_1026, 
      /*output*/ from_t1_offset_1024_to_t1_iter1_pe_1, 
      /*output*/ from_t1_offset_1024_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_992_to_t1_offset_1024);
  Module4Func(
      /*output*/ from_t1_offset_1023_to_t1_offset_1025, 
      /*output*/ from_t1_offset_1023_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_993_to_t1_offset_1023);
  Module5Func(
      /*output*/ from_t1_offset_1026_to_t1_offset_1056, 
      /*output*/ from_t1_offset_1026_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1024_to_t1_offset_1026);
  Module6Func(
      /*output*/ from_t1_offset_1025_to_t1_offset_1057, 
      /*output*/ from_t1_offset_1025_to_t1_iter1_pe_1, 
      /*output*/ from_t1_offset_1025_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1023_to_t1_offset_1025);
  Module4Func(
      /*output*/ from_t1_offset_1056_to_t1_offset_2048, 
      /*output*/ from_t1_offset_1056_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_1026_to_t1_offset_1056);
  Module7Func(
      /*output*/ from_t1_offset_1057_to_t1_offset_2049, 
      /*output*/ from_t1_offset_1057_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1025_to_t1_offset_1057);
  Module8Func(
      /*output*/ from_t1_offset_2048_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_1056_to_t1_offset_2048);
  Module9Func(
      /*output*/ from_t1_iter1_pe_1_to_t1_iter1_offset_0, 
      /* input*/ from_t1_offset_1024_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_1023_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_1025_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_992_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_1056_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_0_to_t1_iter1_pe_1, 
      /* input*/ from_t1_offset_2048_to_t1_iter1_pe_1);
  Module1Func(
      /*output*/ from_t1_iter1_offset_0_to_t1_iter1_offset_992, 
      /*output*/ from_t1_iter1_offset_0_to_t0_pe_1, 
      /* input*/ from_t1_iter1_pe_1_to_t1_iter1_offset_0);
  Module2Func(
      /*output*/ from_t1_iter1_offset_992_to_t1_iter1_offset_1024, 
      /*output*/ from_t1_iter1_offset_992_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_0_to_t1_iter1_offset_992);
  Module3Func(
      /*output*/ from_t1_iter1_offset_1024_to_t1_iter1_offset_1026, 
      /*output*/ from_t1_iter1_offset_1024_to_t0_pe_1, 
      /*output*/ from_t1_iter1_offset_1024_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_992_to_t1_iter1_offset_1024);
  Module5Func(
      /*output*/ from_t1_iter1_offset_1026_to_t1_iter1_offset_1056, 
      /*output*/ from_t1_iter1_offset_1026_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1024_to_t1_iter1_offset_1026);
  Module8Func(
      /*output*/ from_t1_offset_2049_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1057_to_t1_offset_2049);
  Module9Func(
      /*output*/ from_t1_iter1_pe_0_to_t1_iter1_offset_1, 
      /* input*/ from_t1_offset_1025_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1024_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1026_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_993_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1057_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_1_to_t1_iter1_pe_0, 
      /* input*/ from_t1_offset_2049_to_t1_iter1_pe_0);
  Module1Func(
      /*output*/ from_t1_iter1_offset_1_to_t1_iter1_offset_993, 
      /*output*/ from_t1_iter1_offset_1_to_t0_pe_0, 
      /* input*/ from_t1_iter1_pe_0_to_t1_iter1_offset_1);
  Module2Func(
      /*output*/ from_t1_iter1_offset_993_to_t1_iter1_offset_1023, 
      /*output*/ from_t1_iter1_offset_993_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1_to_t1_iter1_offset_993);
  Module4Func(
      /*output*/ from_t1_iter1_offset_1023_to_t1_iter1_offset_1025, 
      /*output*/ from_t1_iter1_offset_1023_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_993_to_t1_iter1_offset_1023);
  Module6Func(
      /*output*/ from_t1_iter1_offset_1025_to_t1_iter1_offset_1057, 
      /*output*/ from_t1_iter1_offset_1025_to_t0_pe_1, 
      /*output*/ from_t1_iter1_offset_1025_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1023_to_t1_iter1_offset_1025);
  Module4Func(
      /*output*/ from_t1_iter1_offset_1056_to_t1_iter1_offset_2048, 
      /*output*/ from_t1_iter1_offset_1056_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_1026_to_t1_iter1_offset_1056);
  Module7Func(
      /*output*/ from_t1_iter1_offset_1057_to_t1_iter1_offset_2049, 
      /*output*/ from_t1_iter1_offset_1057_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1025_to_t1_iter1_offset_1057);
  Module8Func(
      /*output*/ from_t1_iter1_offset_2048_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_1056_to_t1_iter1_offset_2048);
  Module9Func(
      /*output*/ from_t0_pe_1_to_t0_bank_1, 
      /* input*/ from_t1_iter1_offset_1024_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_1023_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_1025_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_992_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_1056_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_0_to_t0_pe_1, 
      /* input*/ from_t1_iter1_offset_2048_to_t0_pe_1);
  Module8Func(
      /*output*/ from_t1_iter1_offset_2049_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1057_to_t1_iter1_offset_2049);
  Module9Func(
      /*output*/ from_t0_pe_0_to_t0_bank_1, 
      /* input*/ from_t1_iter1_offset_1025_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1024_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1026_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_993_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1057_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_1_to_t0_pe_0, 
      /* input*/ from_t1_iter1_offset_2049_to_t0_pe_0);
  Module10Func(
      /*output*/ bank_1_t0_buf, 
      /* input*/ from_t0_pe_1_to_t0_bank_1, 
      /* input*/ from_t0_pe_0_to_t0_bank_1);
  BurstWrite(bank_1_t0, bank_1_t0_buf, coalesced_data_num);
}

}  // extern "C"
