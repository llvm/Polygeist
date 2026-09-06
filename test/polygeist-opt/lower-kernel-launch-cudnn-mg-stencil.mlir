// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnStencil3DSymmetric_f64_memref(
      %input: memref<?x?xf64>, %addend: memref<?x?xf64>,
      %output: memref<?x?xf64>,
      %center: f64, %face: f64, %edge: f64, %corner: f64,
      %alpha: f64, %beta: f64,
      %inD: i32, %inH: i32, %inW: i32,
      %outD: i32, %outH: i32, %outW: i32,
      %strideD: i32, %strideH: i32, %strideW: i32,
      %inOffD: i32, %inOffH: i32, %inOffW: i32,
      %outOffD: i32, %outOffH: i32, %outOffW: i32) {
    kernel.yield
  }

  func.func @run(
      %input: memref<?x?xf64>, %addend: memref<?x?xf64>,
      %output: memref<?x?xf64>,
      %center: f64, %face: f64, %edge: f64, %corner: f64,
      %alpha: f64, %beta: f64,
      %inD: i32, %inH: i32, %inW: i32,
      %outD: i32, %outH: i32, %outW: i32,
      %strideD: i32, %strideH: i32, %strideW: i32,
      %inOffD: i32, %inOffH: i32, %inOffW: i32,
      %outOffD: i32, %outOffH: i32, %outOffW: i32) {
    kernel.launch @cudnnStencil3DSymmetric_f64_memref(
        %input, %addend, %output, %center, %face, %edge, %corner,
        %alpha, %beta, %inD, %inH, %inW, %outD, %outH, %outW,
        %strideD, %strideH, %strideW, %inOffD, %inOffH, %inOffW,
        %outOffD, %outOffH, %outOffW) :
        (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>,
         f64, f64, f64, f64, f64, f64,
         i32, i32, i32, i32, i32, i32, i32, i32, i32,
         i32, i32, i32, i32, i32, i32) -> ()
    return
  }

  func.func @run_erased_pointer(
      %opaque: memref<?xi8>, %center: f64,
      %inD: i32, %inH: i32, %inW: i32) {
    %raw = "polygeist.memref2pointer"(%opaque) : (memref<?xi8>) -> !llvm.ptr
    %view = "polygeist.pointer2memref"(%raw) : (!llvm.ptr) -> memref<?x?xf64>
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %zero_i32 = arith.constant 0 : i32
    %one_i32 = arith.constant 1 : i32
    kernel.launch @cudnnStencil3DSymmetric_f64_memref(
        %view, %view, %view, %center, %zero, %zero, %zero,
        %one, %zero, %inD, %inH, %inW, %inD, %inH, %inW,
        %one_i32, %one_i32, %one_i32,
        %zero_i32, %zero_i32, %zero_i32,
        %zero_i32, %zero_i32, %zero_i32) :
        (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>,
         f64, f64, f64, f64, f64, f64,
         i32, i32, i32, i32, i32, i32, i32, i32, i32,
         i32, i32, i32, i32, i32, i32) -> ()
    return
  }
}

// CHECK-LABEL: func.func @run
// Fifteen dimensions, then six scalars, then three raw pointers.
// CHECK: call @polygeist_cudnn_stencil3d_symmetric_f64(
// CHECK-SAME: %arg9, %arg10, %arg11, %arg12, %arg13, %arg14,
// CHECK-SAME: %arg15, %arg16, %arg17, %arg18, %arg19, %arg20,
// CHECK-SAME: %arg21, %arg22, %arg23,
// CHECK-SAME: %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @run_erased_pointer
// CHECK: %[[RAW:.+]] = "polygeist.memref2pointer"
// CHECK: call @polygeist_cudnn_stencil3d_symmetric_f64(
// CHECK-SAME: %[[RAW]], %[[RAW]], %[[RAW]])
// CHECK: func.func private @polygeist_cudnn_stencil3d_symmetric_f64
