// RUN: mlir-opt --canonicalize %s | FileCheck %s

// CHECK: @kernel_gemm
func private @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    // CHECK: scf.for
    %0 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
      %1 = cmpi "slt", %arg8, %arg0 : i32
      scf.condition(%1) %arg8 : i32
    } do {
    ^bb0(%arg8: i32):  // no predecessors
      %1 = index_cast %arg8 : i32 to index
      // CHECK: scf.for
      %2 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
        %5 = cmpi "slt", %arg9, %arg1 : i32
        scf.condition(%5) %arg9 : i32
      } do {
      ^bb0(%arg9: i32):  // no predecessors
        %5 = index_cast %arg9 : i32 to index
        %6 = load %arg5[%1, %5] : memref<1000x1100xf64>
        %7 = mulf %6, %arg4 : f64
        store %7, %arg5[%1, %5] : memref<1000x1100xf64>
        %8 = addi %arg9, %c1_i32 : i32
        scf.yield %8 : i32
      }
      // CHECK: scf.for
      %3 = scf.while (%arg9 = %2) : (i32) -> i32 {
        %5 = cmpi "slt", %arg9, %arg2 : i32
        scf.condition(%5) %arg9 : i32
      } do {
      ^bb0(%arg9: i32):  // no predecessors
        %5 = index_cast %arg9 : i32 to index
        // CHECK: scf.for
        %6 = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
          %8 = cmpi "slt", %arg10, %arg1 : i32
          scf.condition(%8) %arg10 : i32
        } do {
        ^bb0(%arg10: i32):  // no predecessors
          %8 = index_cast %arg10 : i32 to index
          %9 = load %arg6[%1, %5] : memref<1000x1200xf64>
          %10 = mulf %arg3, %9 : f64
          %11 = load %arg7[%5, %8] : memref<1200x1100xf64>
          %12 = mulf %10, %11 : f64
          %13 = load %arg5[%1, %8] : memref<1000x1100xf64>
          %14 = addf %13, %12 : f64
          store %14, %arg5[%1, %8] : memref<1000x1100xf64>
          %15 = addi %arg10, %c1_i32 : i32
          scf.yield %15 : i32
        }
        %7 = addi %arg9, %c1_i32 : i32
        scf.yield %7 : i32
      }
      %4 = addi %arg8, %c1_i32 : i32
      scf.yield %4 : i32
    }
    return
}
