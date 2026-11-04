// A member of hls::stream stays an external call in a compile of several
// translation units. An earlier unit's external call to `operator<<` put its
// name on the list of functions to emit when found; the later units that use
// it instantiate its definition, and cgeist then emitted the synthesis body
// (`write`, whose own body calls __fpga_fifo_push) and inlined it (CIRCT
// OPEN_ISSUES O193: flowgnn/dgn, three units). With one unit, or with the
// fix, every member is an external declaration.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %S/Inputs/stream-member-cross-unit-b.cpp %S/Inputs/stream-member-cross-unit-c.cpp %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.abc.mlir
// RUN: FileCheck %s < %t.abc.mlir
// RUN: FileCheck %s --check-prefix=NOFIFO < %t.abc.mlir
// RUN: cgeist %S/Inputs/stream-member-cross-unit-c.cpp %s %S/Inputs/stream-member-cross-unit-b.cpp %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.cab.mlir
// RUN: FileCheck %s < %t.cab.mlir
// RUN: FileCheck %s --check-prefix=NOFIFO < %t.cab.mlir
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=ONE

// NOFIFO-NOT: __fpga_fifo
// CHECK-DAG: func.func @_Z4prodP{{.*}}(
// CHECK-DAG: func.func @_Z5again{{.*}}(
// CHECK-DAG: func.func private @{{.*}}lsERKS6_(
// CHECK-DAG: func.func private @{{.*}}5writeERKS6_(
// ONE: func.func @_Z4prodP{{.*}}(
// ONE: call @{{.*}}lsERKS6_(
// ONE: func.func private @{{.*}}lsERKS6_(
#include <hls_stream.h>
#include <ap_fixed.h>
#include <array>
typedef std::array<ap_fixed<16, 3>, 2> tok_t;
void prod(tok_t *in, hls::stream<tok_t> &s) {
  for (int i = 0; i < 4; i++)
    s << in[i];
}
