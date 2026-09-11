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
  kernel.defn @cublasSsyrk(
      %a: tensor<?x?xf32>, %c: tensor<?x?xf32>, %beta: f32, %alpha: f32)
      -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }
  kernel.defn @cublasSsyr2k(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
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

  func.func @ssyrk_slice(%a: tensor<?x?xf32>, %c: memref<?x1200xf32>,
                         %n: index, %beta: f32, %alpha: f32) {
    %ct = bufferization.to_tensor %c : memref<?x1200xf32>
    %cv = tensor.extract_slice %ct[0, 0] [%n, %n] [1, 1]
      : tensor<?x1200xf32> to tensor<?x?xf32>
    %r = kernel.launch @cublasSsyrk(%a, %cv, %beta, %alpha) {
      polygeist.contraction.backend = "cublas-tf32"
    } : (tensor<?x?xf32>, tensor<?x?xf32>, f32, f32) -> tensor<?x?xf32>
    %full = tensor.insert_slice %r into %ct[0, 0] [%n, %n] [1, 1]
      : tensor<?x?xf32> into tensor<?x1200xf32>
    %out = bufferization.to_memref %full : memref<?x1200xf32>
    memref.copy %out, %c : memref<?x1200xf32> to memref<?x1200xf32>
    return
  }

  func.func @ssyr2k(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                    %c: tensor<?x?xf32>, %beta: f32, %alpha: f32)
      -> tensor<?x?xf32> {
    %r = kernel.launch @cublasSsyr2k(%a, %b, %c, %beta, %alpha) {
      polygeist.contraction.backend = "cublas-tf32"
    } : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32, f32)
          -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @syrk
// CHECK: call @polygeist_cublas_dsyrk_lower
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @syr2k
// CHECK: call @polygeist_cublas_dsyr2k_lower
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @ssyrk_slice
// CHECK: call @polygeist_cublas_ssyrk_lower_tf32
// CHECK-NOT: tensor.insert_slice
// CHECK-LABEL: func.func @ssyr2k
// CHECK: call @polygeist_cublas_ssyr2k_lower_tf32
// CHECK-NOT: kernel.launch
