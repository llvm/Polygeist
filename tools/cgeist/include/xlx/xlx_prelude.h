//===- xlx_prelude.h - Xilinx HLS builtins for cgeist ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Force-included by cgeist (after the Vitis HLS etc/autopilot_ssdm_op.h) when
// it compiles against the real Vitis HLS headers (the harness's `--shim
// real`). Declares the Xilinx HLS builtins the headers call under
// __SYNTHESIS__ so that upstream clang parses them; cgeist lowers every call
// by name (Lib/CGXlxBuiltin.cc). Written for Polygeist; the signatures follow
// hls-llvm-project ee930469 clang/include/clang/Basic/Builtins.def:1464-1475
// ("vvC*vC*vC*" etc.) and BuiltinsFPGA.def:145 (__fpga_ctlz), with `const`
// dropped from the result pointer so that `(void*)&x` binds without a cast.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_XLX_PRELUDE_H
#define CGEIST_XLX_PRELUDE_H

#ifdef __cplusplus
extern "C" {
#endif
void __builtin_bit_concat(void *, const void *, const void *);
void __builtin_bit_from_string(void *, const char *, int);
int __builtin_bit_select(const void *, int);
void __builtin_bit_set(void *, const void *, const void *, int);
void __builtin_bit_part_select(void *, const void *, int, int);
void __builtin_bit_part_set(void *, const void *, const void *, int, int);
int __builtin_bit_and_reduce(const void *);
int __builtin_bit_or_reduce(const void *);
int __builtin_bit_nxor_reduce(const void *);
int __builtin_bit_nand_reduce(const void *);
int __builtin_bit_nor_reduce(const void *);
int __builtin_bit_xor_reduce(const void *);
#ifdef __cplusplus
// hls::stream (hls_stream_39.h) and its AXI-Stream specialisation
// (ap_axi_sdata.h); BuiltinsFPGA.def:70-77 and 108-113. Declared only so that
// the headers' member bodies parse: under --xlx-headers cgeist never emits a
// member of hls::stream (each call is an external call, as with the stand-in
// headers; clang-mlir.cc isXlxStreamClass), so these are never lowered.
bool __fpga_fifo_not_empty(const void *);
bool __fpga_fifo_not_full(const void *);
bool __fpga_fifo_nb_pop(void *, void *);
bool __fpga_fifo_nb_push(void *, const void *);
void __fpga_fifo_pop(void *, void *);
void __fpga_fifo_push(void *, const void *);
int __fpga_fifo_size(const void *);
int __fpga_fifo_capacity(const void *);
bool __fpga_axis_ready(const void *, const void *, const void *, const void *,
                       const void *, const void *, const void *);
bool __fpga_axis_valid(const void *, const void *, const void *, const void *,
                       const void *, const void *, const void *);
void __fpga_axis_pop(void *, void *, void *, void *, void *, void *, void *,
                     void *, void *, void *, void *, void *, void *, void *);
bool __fpga_axis_push(void *, void *, void *, void *, void *, void *, void *,
                      const void *, const void *, const void *, const void *,
                      const void *, const void *, const void *);
bool __fpga_axis_nb_pop(void *, void *, void *, void *, void *, void *, void *,
                        void *, void *, void *, void *, void *, void *, void *);
bool __fpga_axis_nb_push(void *, void *, void *, void *, void *, void *,
                         void *, const void *, const void *, const void *,
                         const void *, const void *, const void *,
                         const void *);
#endif
#ifdef __cplusplus
}

// Vitis declares these as generic "count at the operand's width" builtins;
// a template keeps the operand's own type (a _BitInt) at the call.
template <typename T> int __fpga_ctlz(T);
template <typename T> int __fpga_cttz(T);
// ap_shift_reg (ap_shift_reg.h): read the element at an address, and shift
// the register by one place (cgeist lowers both by name, CGXlxBuiltin.cc).
template <typename T> T __fpga_shift_register_peek(T *, unsigned int);
template <typename T> void __fpga_shift_register_shift(T, T *, bool);
#endif

// Vitis 2025.1 compiles with the mingw sysroot of its Windows install, whose
// standard headers declare more than glibc / libstdc++ do: <iostream> and
// <cstdlib> bring ::memcpy, <hls_vector.h> ::uint32_t (CIRCT OPEN_ISSUES
// O76: class_ece8893 oscar_baglivio lab_2 / lab_3, vitis_examples
// streaming_vadd compile in Vitis and fail here). The two names are declared
// as glibc declares them, so a later <string.h> / <stdint.h> agrees.
#include <stdint.h>
#ifdef __cplusplus
extern "C" void *memcpy(void *__restrict, const void *__restrict,
                        __SIZE_TYPE__) noexcept(true);
#endif

// hls_task.h (2025.1) marks the call of an hls::task's function with the
// Vitis statement attribute `xlx_infinite_task(&tag)`, which clang does not
// know: an unknown GNU attribute in front of an expression statement is
// parsed as the start of a declaration ("expected unqualified-id"). The call
// itself is what the dataflow region runs; clang's own call-statement
// attribute nomerge stands in, without effect on the code (CIRCT OPEN_ISSUES
// O76: class_ece8893 ben_hall lab3).
#define xlx_infinite_task(...) nomerge

#endif // CGEIST_XLX_PRELUDE_H
