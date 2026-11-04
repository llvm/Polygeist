// hls_task.h (2025.1) marks the call of an hls::task's function with the
// Vitis statement attribute xlx_infinite_task(&tag), which clang does not
// know; the prelude spells it as clang's call-statement attribute nomerge, so
// the task's function is called (and inlined) in the dataflow region like a
// process (CIRCT OPEN_ISSUES O76: ben_hall lab3).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <hls_stream.h>
#include <hls_task.h>

// CHECK-LABEL: func.func @_Z5c_topRN3hls6streamIiLi0EEES2_
// CHECK-SAME: hls.dataflow
// CHECK: call @_ZN3hls6streamIiLi0EE4readEv(%arg0)
// CHECK: call @_ZN3hls6streamIiLi0EE5writeERKi
// CHECK: call @_ZN3hls6streamIiLi0EE4readEv
// CHECK: call @_ZN3hls6streamIiLi0EE5writeERKi(%arg1
static void inc(hls::stream<int> &i, hls::stream<int> &o) {
  o.write(i.read() + 1);
}
void c_top(hls::stream<int> &in, hls::stream<int> &out) {
#pragma HLS dataflow
  hls_thread_local hls::stream<int> mid;
  hls_thread_local hls::task t(inc, in, mid);
  hls_thread_local hls::task t2(inc, mid, out);
}
