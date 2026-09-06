// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cusparseXcoo2csr_i32_memref(
      %rows: index, %coo: memref<?xi32>, %csr: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cusparseXcsr2coo_i32_memref(
      %rows: index, %csr: memref<?xi32>, %coo: memref<?xi32>) {
    kernel.yield
  }
  func.func @conversions(%rows: index, %coo: memref<?xi32>,
                         %csr: memref<?xi32>, %roundtrip: memref<?xi32>) {
    kernel.launch @cusparseXcoo2csr_i32_memref(%rows, %coo, %csr) :
        (index, memref<?xi32>, memref<?xi32>) -> ()
    kernel.launch @cusparseXcsr2coo_i32_memref(%rows, %csr, %roundtrip) :
        (index, memref<?xi32>, memref<?xi32>) -> ()
    return
  }
}

// CHECK: call @polygeist_cusparse_coo2csr_i32_sized
// CHECK: call @polygeist_cusparse_csr2coo_i32_sized
// CHECK-NOT: kernel.launch
