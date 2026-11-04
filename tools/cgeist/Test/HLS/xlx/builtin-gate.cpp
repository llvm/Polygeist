// The Xilinx HLS bit builtins are lowered to arith only under --xlx-prelude
// (which declares them) or --xlx-headers. Without either option a function of
// the same name is an ordinary external call, as before track W11 (W11 review
// D3).
// RUN: cgeist %s --function='*' -S -o - | FileCheck %s --check-prefix=OFF
// RUN: cgeist %s --xlx-headers --function='*' -S -o - | FileCheck %s --check-prefix=ON

extern "C" int __fpga_ctlz(unsigned x);

// OFF-LABEL: func.func @clz(
// OFF: call @__fpga_ctlz
// ON-LABEL: func.func @clz(
// ON-NOT: call @__fpga_ctlz
// ON: math.ctlz
extern "C" int clz(unsigned x) { return __fpga_ctlz(x); }
