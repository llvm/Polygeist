// Console output through the C++ streams is no hardware: Vitis' reflow
// removes it (class_ece8893_2023_spring YOLOv4-Tiny Maxpool_Layer prints a
// progress line from the top function; its a.o.1.ll has no ostream call
// left). cgeist used to emit the calls with the libstdc++ internals they
// reach, whose `llvm.load` of a memref does not verify outside Polygeist
// (hls-est: "invalid memref element type", W8b). The calls are dropped; an
// argument with a side effect is still evaluated.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <iostream>

// CHECK-LABEL: func.func @top(
// CHECK-NOT:     call
// CHECK-NOT:     polygeist.get_func
// CHECK:         affine.store
// CHECK-NOT:     call
// CHECK:         return
// CHECK-NOT:     basic_ostream
extern "C" void top(int a[8], int n) {
  for (int i = 0; i < 8; i++) {
    std::cout << "tile " << i << "/" << n << std::endl;
    a[i] = a[i] + n;
  }
  std::cerr.flush();
}

// The counter's increment inside the printed expression is kept.
// CHECK-LABEL: func.func @side_effect(
// CHECK:         %[[K:.*]] = affine.load
// CHECK:         %[[K1:.*]] = arith.addi %[[K]]
// CHECK:         affine.store %[[K1]]
// CHECK-NOT:     call
extern "C" void side_effect(int c[1]) {
  int k = c[0];
  std::cout << k++ << '\n';
  c[0] = k;
}
