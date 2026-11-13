// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnConvolution2DWindow_f32(
      %input: tensor<?x?x?x?xf32>, %output: tensor<?x?x?x?xf32>,
      %weight: f32,
      %kh: i32, %kw: i32, %sh: i32, %sw: i32,
      %dh: i32, %dw: i32, %ph: i32, %pw: i32)
      -> tensor<?x?x?x?xf32> {
    kernel.yield %output : tensor<?x?x?x?xf32>
  }

  func.func @uniform_window(
      %input: tensor<?x?x?x?xf32>, %output: tensor<?x?x?x?xf32>,
      %weight: f32) -> tensor<?x?x?x?xf32> {
    %kh = arith.constant 2 : i32
    %kw = arith.constant 3 : i32
    %sh = arith.constant 2 : i32
    %sw = arith.constant 1 : i32
    %dh = arith.constant 1 : i32
    %dw = arith.constant 2 : i32
    %ph = arith.constant 0 : i32
    %pw = arith.constant 1 : i32
    %result = kernel.launch @cudnnConvolution2DWindow_f32(
        %input, %output, %weight,
        %kh, %kw, %sh, %sw, %dh, %dw, %ph, %pw)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32,
           i32, i32, i32, i32, i32, i32, i32, i32)
          -> tensor<?x?x?x?xf32>
    return %result : tensor<?x?x?x?xf32>
  }
}

// CHECK-LABEL: func.func @uniform_window
// CHECK: call @polygeist_cudnn_conv2d_uniform_window_f32(
// CHECK-SAME: i32, i32, i32, i32, i32, i32, f32,
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr
// CHECK-NOT: kernel.launch
