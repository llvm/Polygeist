// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasDsyrk(
      %a: tensor<?x?xf64>, %c: tensor<?x?xf64>, %beta: f64, %alpha: f64)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }
  kernel.defn @cublasDsyr2k(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  func.func @syrk(%a: tensor<?x?xf64>, %c: tensor<?x?xf64>,
                  %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %r = kernel.launch @cublasDsyrk(%a, %c, %beta, %alpha)
        : (tensor<?x?xf64>, tensor<?x?xf64>, f64, f64) -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }

  func.func @syr2k(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>,
                   %c: tensor<?x?xf64>, %beta: f64, %alpha: f64)
      -> tensor<?x?xf64> {
    %r = kernel.launch @cublasDsyr2k(%a, %b, %c, %beta, %alpha)
        : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, f64, f64)
          -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @syrk
// CHECK: call @polygeist_cublas_dsyrk_lower
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @syr2k
// CHECK: call @polygeist_cublas_dsyr2k_lower
// CHECK-NOT: kernel.launch
