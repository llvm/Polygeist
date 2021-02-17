// RUN: mlir-opt --canonicalize %s | FileCheck %s

// CHECK: @kernel_gemm
func private @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %0 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
      %1 = cmpi "slt", %arg8, %arg0 : i32
      scf.condition(%1) %arg8 : i32
    } do {
    ^bb0(%arg8: i32):  // no predecessors
      %1 = index_cast %arg8 : i32 to index
      // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
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
      // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
      %3 = scf.while (%arg9 = %2) : (i32) -> i32 {
        %5 = cmpi "slt", %arg9, %arg2 : i32
        scf.condition(%5) %arg9 : i32
      } do {
      ^bb0(%arg9: i32):  // no predecessors
        %5 = index_cast %arg9 : i32 to index
        // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
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

// CHECK: @kernel_gemver
func private @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %0:2 = scf.while (%arg12 = %c0_i32) : (i32) -> (i32, i32) {
      %4 = cmpi "slt", %arg12, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg12 : i32, i32
    } do {
    ^bb0(%arg12: i32, %arg13: i32):  // no predecessors
      %4 = index_cast %arg13 : i32 to index
      // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
      %5 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi "slt", %arg14, %arg0 : i32
        scf.condition(%7) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):  // no predecessors
        %7 = index_cast %arg14 : i32 to index
        %8 = load %arg3[%4, %7] : memref<2000x2000xf64>
        %9 = load %arg4[%4] : memref<2000xf64>
        %10 = load %arg5[%7] : memref<2000xf64>
        %11 = mulf %9, %10 : f64
        %12 = addf %8, %11 : f64
        %13 = load %arg6[%4] : memref<2000xf64>
        %14 = load %arg7[%7] : memref<2000xf64>
        %15 = mulf %13, %14 : f64
        %16 = addf %12, %15 : f64
        store %16, %arg3[%4, %7] : memref<2000x2000xf64>
        %17 = addi %arg14, %c1_i32 : i32
        scf.yield %17 : i32
      }
      %6 = addi %arg13, %c1_i32 : i32
      scf.yield %6 : i32
    }
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %1:2 = scf.while (%arg12 = %0#0) : (i32) -> (i32, i32) {
      %4 = cmpi "slt", %arg12, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg12 : i32, i32
    } do {
    ^bb0(%arg12: i32, %arg13: i32):  // no predecessors
      %4 = index_cast %arg13 : i32 to index
      // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
      %5 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi "slt", %arg14, %arg0 : i32
        scf.condition(%7) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):  // no predecessors
        %7 = index_cast %arg14 : i32 to index
        %8 = load %arg9[%4] : memref<2000xf64>
        %9 = load %arg3[%7, %4] : memref<2000x2000xf64>
        %10 = mulf %arg2, %9 : f64
        %11 = load %arg10[%7] : memref<2000xf64>
        %12 = mulf %10, %11 : f64
        %13 = addf %8, %12 : f64
        store %13, %arg9[%4] : memref<2000xf64>
        %14 = addi %arg14, %c1_i32 : i32
        scf.yield %14 : i32
      }
      %6 = addi %arg13, %c1_i32 : i32
      scf.yield %6 : i32
    }
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %2:2 = scf.while (%arg12 = %1#0) : (i32) -> (i32, i32) {
      %4 = cmpi "slt", %arg12, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg12 : i32, i32
    } do {
    ^bb0(%arg12: i32, %arg13: i32):  // no predecessors
      %4 = index_cast %arg13 : i32 to index
      %5 = load %arg9[%4] : memref<2000xf64>
      %6 = load %arg11[%4] : memref<2000xf64>
      %7 = addf %5, %6 : f64
      store %7, %arg9[%4] : memref<2000xf64>
      %8 = addi %arg13, %c1_i32 : i32
      scf.yield %8 : i32
    }
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %3 = scf.while (%arg12 = %2#0) : (i32) -> i32 {
      %4 = cmpi "slt", %arg12, %arg0 : i32
      scf.condition(%4) %arg12 : i32
    } do {
    ^bb0(%arg12: i32):  // no predecessors
      %4 = index_cast %arg12 : i32 to index
      %5 = scf.while (%arg13 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi "slt", %arg13, %arg0 : i32
        scf.condition(%7) %arg13 : i32
      } do {
      ^bb0(%arg13: i32):  // no predecessors
        %7 = index_cast %arg13 : i32 to index
        %8 = load %arg8[%4] : memref<2000xf64>
        %9 = load %arg3[%4, %7] : memref<2000x2000xf64>
        %10 = mulf %arg1, %9 : f64
        %11 = load %arg9[%7] : memref<2000xf64>
        %12 = mulf %10, %11 : f64
        %13 = addf %8, %12 : f64
        store %13, %arg8[%4] : memref<2000xf64>
        %14 = addi %arg13, %c1_i32 : i32
        scf.yield %14 : i32
      }
      %6 = addi %arg12, %c1_i32 : i32
      scf.yield %6 : i32
    }
    return
  }

// CHECK: @kernel_symm
func private @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    %0 = scf.while (%arg7 = %c0_i32) : (i32) -> i32 {
      %1 = cmpi "slt", %arg7, %arg0 : i32
      scf.condition(%1) %arg7 : i32
    } do {
    ^bb0(%arg7: i32):  // no predecessors
      %1 = index_cast %arg7 : i32 to index
      // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
      %2 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
        %4 = cmpi "slt", %arg8, %arg1 : i32
        scf.condition(%4) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):  // no predecessors
        %4 = index_cast %arg8 : i32 to index
        %5 = sitofp %c0_i32 : i32 to f64
        // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
        %6:2 = scf.while (%arg9 = %c0_i32, %arg10 = %5) : (i32, f64) -> (f64, i32) {
          %17 = cmpi "slt", %arg9, %arg7 : i32
          scf.condition(%17) %arg10, %arg9 : f64, i32
        } do {
        ^bb0(%arg9: f64, %arg10: i32):  // no predecessors
          %17 = index_cast %arg10 : i32 to index
          %18 = load %arg6[%1, %4] : memref<1000x1200xf64>
          %19 = mulf %arg2, %18 : f64
          %20 = load %arg5[%1, %17] : memref<1000x1000xf64>
          %21 = mulf %19, %20 : f64
          %22 = load %arg4[%17, %4] : memref<1000x1200xf64>
          %23 = addf %22, %21 : f64
          store %23, %arg4[%17, %4] : memref<1000x1200xf64>
          %24 = load %arg6[%17, %4] : memref<1000x1200xf64>
          %25 = load %arg5[%1, %17] : memref<1000x1000xf64>
          %26 = mulf %24, %25 : f64
          %27 = addf %arg9, %26 : f64
          %28 = addi %arg10, %c1_i32 : i32
          scf.yield %28, %27 : i32, f64
        }
        %7 = load %arg4[%1, %4] : memref<1000x1200xf64>
        %8 = mulf %arg3, %7 : f64
        %9 = load %arg6[%1, %4] : memref<1000x1200xf64>
        %10 = mulf %arg2, %9 : f64
        %11 = load %arg5[%1, %1] : memref<1000x1000xf64>
        %12 = mulf %10, %11 : f64
        %13 = addf %8, %12 : f64
        %14 = mulf %arg2, %6#0 : f64
        %15 = addf %13, %14 : f64
        store %15, %arg4[%1, %4] : memref<1000x1200xf64>
        %16 = addi %arg8, %c1_i32 : i32
        scf.yield %16 : i32
      }
      %3 = addi %arg7, %c1_i32 : i32
      scf.yield %3 : i32
    }
    return
  }
