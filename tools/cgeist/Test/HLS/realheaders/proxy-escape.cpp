// A proxy object that cannot be scalarised (here an ap_range_ref passed by
// value to a function that is not inlined) would leave a memref of an LLVM
// struct holding a memref, which is not valid MLIR once printed. Under
// --xlx-headers cgeist reports it and fails (exit code 13) instead of printing
// it (W11 review D1).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: not cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o /dev/null 2>&1 | FileCheck %s
#include <ap_int.h>

// CHECK: error: a Vitis HLS proxy object (ap_range_ref, ap_bit_ref, ap_concat_ref, ...) could not be scalarised
__attribute__((noinline)) unsigned f(ap_range_ref<32, false> r) {
  return r.to_uint();
}
extern "C" unsigned top(ap_uint<32> x) { return f(x.range(7, 0)); }
