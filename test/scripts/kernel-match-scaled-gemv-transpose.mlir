// RUN: polygeist-opt %s -o %t.input.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.input.mlir > %t
// RUN: FileCheck %s < %t

module {
  // CHECK-LABEL: func.func @arbitrary_transpose
  // CHECK: kernel.launch @cublasDgemv_alpha_T(
  func.func @arbitrary_transpose(%a: tensor<17x19xf64>, %x: tensor<17xf64>,
                                %y: tensor<19xf64>, %alpha: f64) -> tensor<19xf64> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d1,d0)>,
                       affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : tensor<17x19xf64>, tensor<17xf64>) outs(%y : tensor<19xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %p = arith.mulf %av, %xv : f64
      %s = arith.mulf %alpha, %p : f64
      %v = arith.addf %out, %s : f64
      linalg.yield %v : f64
    } -> tensor<19xf64>
    return %r : tensor<19xf64>
  }

  // CHECK-LABEL: func.func @arbitrary_forward
  // CHECK: kernel.launch @cublasDgemv_alpha(
  func.func @arbitrary_forward(%a: tensor<17x19xf64>, %x: tensor<19xf64>,
                              %y: tensor<17xf64>, %alpha: f64) -> tensor<17xf64> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                       affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : tensor<17x19xf64>, tensor<19xf64>) outs(%y : tensor<17xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %p = arith.mulf %av, %xv : f64
      %s = arith.mulf %alpha, %p : f64
      %v = arith.addf %out, %s : f64
      linalg.yield %v : f64
    } -> tensor<17xf64>
    return %r : tensor<17xf64>
  }

  // CHECK-LABEL: func.func @arbitrary_subtract
  // CHECK-NOT: kernel.launch @cublasDgemv_alpha_T(
  // CHECK: linalg.generic
  func.func @arbitrary_subtract(%a: tensor<17x19xf64>, %x: tensor<17xf64>,
                               %y: tensor<19xf64>, %alpha: f64) -> tensor<19xf64> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d1,d0)>,
                       affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : tensor<17x19xf64>, tensor<17xf64>) outs(%y : tensor<19xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %p = arith.mulf %av, %xv : f64
      %s = arith.mulf %alpha, %p : f64
      %v = arith.subf %out, %s : f64
      linalg.yield %v : f64
    } -> tensor<19xf64>
    return %r : tensor<19xf64>
  }
  // CHECK-LABEL: func.func @renamed_float_transpose
  // CHECK: kernel.launch @cublasSgemv_alpha_T(
  func.func @renamed_float_transpose(%a: tensor<17x19xf32>, %x: tensor<17xf32>,
                                %y: tensor<19xf32>, %alpha: f32) -> tensor<19xf32> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d1,d0)>,
                       affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : tensor<17x19xf32>, tensor<17xf32>) outs(%y : tensor<19xf32>) {
    ^bb0(%av: f32, %xv: f32, %out: f32):
      %p = arith.mulf %av, %xv : f32
      %s = arith.mulf %alpha, %p : f32
      %v = arith.addf %out, %s : f32
      linalg.yield %v : f32
    } -> tensor<19xf32>
    return %r : tensor<19xf32>
  }

  // CHECK-LABEL: func.func @renamed_float_forward
  // CHECK: kernel.launch @cublasSgemv_alpha(
  func.func @renamed_float_forward(%a: tensor<17x19xf32>, %x: tensor<19xf32>,
                              %y: tensor<17xf32>, %alpha: f32) -> tensor<17xf32> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                       affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %x : tensor<17x19xf32>, tensor<19xf32>) outs(%y : tensor<17xf32>) {
    ^bb0(%av: f32, %xv: f32, %out: f32):
      %p = arith.mulf %av, %xv : f32
      %s = arith.mulf %alpha, %p : f32
      %v = arith.addf %out, %s : f32
      linalg.yield %v : f32
    } -> tensor<17xf32>
    return %r : tensor<17xf32>
  }

}
