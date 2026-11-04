// A pointer cast to a pointer to an array, `(T (*)[8][8]) p`, has cgeist's
// type for a pointer to an array: the pointee's own memref<8x8xT>. Passed to
// a parameter `T a[4][8][8]` (one rank higher, the same trailing shape) it
// gave a func.call whose operand type did not match the callee
// (class_ece8893_2023_spring GradCAM src_hls, W8b). C addresses the
// parameter through the pointer, row-major from its first element, so the
// argument is re-typed through its address.
// RUN: cgeist %s %resourcedir --function='*' -S -O0 --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=FULL

void callee(short out[4][8][8]) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 8; j++)
      for (int k = 0; k < 8; k++)
        out[i][j][k] = i + j + k;
}

// CHECK-LABEL: func.func @_Z3topPs(
// CHECK:         %[[R:.*]] = "polygeist.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<4x8x8xi16>
// CHECK:         call @_Z6calleePA8_A8_s(%[[R]]) : (memref<4x8x8xi16>) -> ()

// Inlined, the stores address the buffer linearly: out[i][j][k] is
// dram[(i * 8 + j) * 8 + k].
// FULL-LABEL: func.func @_Z3topPs(
// FULL:         llvm.getelementptr
// FULL:         llvm.store
void top(short dram[1024]) {
  short *p = dram;
  callee((short (*)[8][8])p);
}
