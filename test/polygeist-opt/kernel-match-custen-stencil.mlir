// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --stencil-backend=custen 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

#id = affine_map<(i, j) -> (i, j)>
module {
  func.func @stencil(%input: tensor<?x?xf64>, %output: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    %w = arith.constant 1.0 : f64
    %s00 = tensor.extract_slice %input[0, 0] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s01 = tensor.extract_slice %input[0, 1] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s02 = tensor.extract_slice %input[0, 2] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s10 = tensor.extract_slice %input[1, 0] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s11 = tensor.extract_slice %input[1, 1] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s12 = tensor.extract_slice %input[1, 2] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s20 = tensor.extract_slice %input[2, 0] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s21 = tensor.extract_slice %input[2, 1] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %s22 = tensor.extract_slice %input[2, 2] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %dst = tensor.extract_slice %output[1, 1] [3, 3] [1, 1]
        : tensor<?x?xf64> to tensor<?x?xf64>
    %r = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id, #id, #id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%s00, %s01, %s02, %s10, %s11, %s12, %s20, %s21, %s22 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%dst : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %c: f64, %d: f64, %e: f64,
         %f: f64, %g: f64, %h: f64, %i: f64, %out: f64):
      %m0 = arith.mulf %a, %w : f64
      %m1 = arith.mulf %b, %w : f64
      %a1 = arith.addf %m0, %m1 : f64
      %m2 = arith.mulf %c, %w : f64
      %a2 = arith.addf %a1, %m2 : f64
      %m3 = arith.mulf %d, %w : f64
      %a3 = arith.addf %a2, %m3 : f64
      %m4 = arith.mulf %e, %w : f64
      %a4 = arith.addf %a3, %m4 : f64
      %m5 = arith.mulf %f, %w : f64
      %a5 = arith.addf %a4, %m5 : f64
      %m6 = arith.mulf %g, %w : f64
      %a6 = arith.addf %a5, %m6 : f64
      %m7 = arith.mulf %h, %w : f64
      %a7 = arith.addf %a6, %m7 : f64
      %m8 = arith.mulf %i, %w : f64
      %a8 = arith.addf %a7, %m8 : f64
      linalg.yield %a8 : f64
    } -> tensor<?x?xf64>
    %result = tensor.insert_slice %r into %output[1, 1] [3, 3] [1, 1]
        : tensor<?x?xf64> into tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// CHECK: kernel.launch @custenStencil2DXY_f64_tensor
