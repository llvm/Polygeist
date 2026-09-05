// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cusparseSpMV_CSR_f64_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
    kernel.yield
  }
  func.func @spmv(%rows: index, %rowptr: memref<?xi32>,
                  %cols: memref<?xi32>, %values: memref<?xf64>,
                  %x: memref<?xf64>, %y: memref<?xf64>) {
    kernel.launch @cusparseSpMV_CSR_f64_memref(
        %rows, %rowptr, %cols, %values, %x, %y) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf64>,
         memref<?xf64>, memref<?xf64>) -> ()
    return
  }
}

// CHECK: call @polygeist_cusparse_spmv_csr_f64_sized
// CHECK-NOT: kernel.launch
