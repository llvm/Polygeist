// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas --lower-polygeist-submap | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

module {
  kernel.defn @cusparseSpMV_CSR_f64_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMM_CSR_f32_memref(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf32>, %b: memref<?x?xf32>,
      %c: memref<?x?xf32>) { kernel.yield }
  kernel.defn @cusparseSpMM_COO_f32_memref(
      %rows: index, %nnz: index,
      %row_indices: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf32>, %b: memref<?x?xf32>,
      %c: memref<?x?xf32>) { kernel.yield }
  kernel.defn @cusparseSpMM_BSR_f32_memref(
      %block_rows: index, %block_dim: index,
      %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?x?x?xf32>, %x: memref<?xf32>,
      %y: memref<?xf32>) { kernel.yield }
  func.func @spmv(%rows: index, %rowptr: memref<?xi32>,
                  %cols: memref<?xi32>, %values: memref<?xf64>,
                  %x: memref<?xf64>, %y: memref<?xf64>) {
    kernel.launch @cusparseSpMV_CSR_f64_memref(
        %rows, %rowptr, %cols, %values, %x, %y) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf64>,
         memref<?xf64>, memref<?xf64>) -> ()
    return
  }

  // Raised C loops commonly retain logical one-dimensional submaps until ABI
  // lowering. The runtime call must consume their explicit sizes and backing
  // pointers so no Polygeist view operation survives into standard MLIR.
  func.func @spmv_through_submaps(
      %rows: index, %rowptr: memref<?xi32>, %cols: memref<?xi32>,
      %values: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
    %one = arith.constant 1 : index
    %row_count = arith.addi %rows, %one : index
    %row_view = polygeist.submap(%rowptr, %row_count) {map = #identity} :
      (memref<?xi32>, index) -> memref<?xi32>
    %col_view = polygeist.submap(%cols, %rows) {map = #identity} :
      (memref<?xi32>, index) -> memref<?xi32>
    %value_view = polygeist.submap(%values, %rows) {map = #identity} :
      (memref<?xf64>, index) -> memref<?xf64>
    %x_view = polygeist.submap(%x, %rows) {map = #identity} :
      (memref<?xf64>, index) -> memref<?xf64>
    %y_view = polygeist.submap(%y, %rows) {map = #identity} :
      (memref<?xf64>, index) -> memref<?xf64>
    kernel.launch @cusparseSpMV_CSR_f64_memref(
        %rows, %row_view, %col_view, %value_view, %x_view, %y_view) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf64>,
         memref<?xf64>, memref<?xf64>) -> ()
    return
  }

  func.func @spmm(%rows: index, %rowptr: memref<?xi32>,
                  %cols: memref<?xi32>, %values: memref<?xf32>,
                  %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
    kernel.launch @cusparseSpMM_CSR_f32_memref(
        %rows, %rowptr, %cols, %values, %b, %c) :
        (index, memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }

  func.func @coo_spmm(%rows: index, %nnz: index,
                      %row_indices: memref<?xi32>,
                      %cols: memref<?xi32>, %values: memref<?xf32>,
                      %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
    kernel.launch @cusparseSpMM_COO_f32_memref(
        %rows, %nnz, %row_indices, %cols, %values, %b, %c) :
        (index, index, memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }

  func.func @bsr_spmm(%block_rows: index, %block_dim: index,
                      %rowptr: memref<?xi32>, %cols: memref<?xi32>,
                      %values: memref<?x?x?xf32>, %x: memref<?xf32>,
                      %y: memref<?xf32>) {
    kernel.launch @cusparseSpMM_BSR_f32_memref(
        %block_rows, %block_dim, %rowptr, %cols, %values, %x, %y) :
        (index, index, memref<?xi32>, memref<?xi32>, memref<?x?x?xf32>,
         memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK: call @polygeist_cusparse_spmv_csr_f64_sized
// CHECK: call @polygeist_cusparse_spmm_csr_f32_sized
// CHECK: call @polygeist_cusparse_spmm_coo_f32_sized
// CHECK: call @polygeist_cusparse_spmm_bsr_f32_sized
// CHECK-NOT: kernel.launch
// CHECK-NOT: polygeist.submap
