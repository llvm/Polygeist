// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cusparseSpMV_JDS_f32_memref(
      %rows: index, %repetitions: index, %counts: memref<?xi32>,
      %offsets: memref<?xi32>, %columns: memref<?xi32>,
      %values: memref<?xf32>, %permutation: memref<?xi32>,
      %x: memref<?xf32>, %y: memref<?xf32>) { kernel.yield }

  func.func @run(%rows: index, %repetitions: index,
                 %counts: memref<?xi32>, %offsets: memref<?xi32>,
                 %columns: memref<?xi32>, %values: memref<?xf32>,
                 %permutation: memref<?xi32>, %x: memref<?xf32>,
                 %y: memref<?xf32>) {
    kernel.launch @cusparseSpMV_JDS_f32_memref(
        %rows, %repetitions, %counts, %offsets, %columns, %values,
        %permutation, %x, %y) :
        (index, index, memref<?xi32>, memref<?xi32>, memref<?xi32>,
         memref<?xf32>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @run
// CHECK: %[[ROWS:.+]] = arith.index_cast %arg0 : index to i32
// CHECK: %[[REPETITIONS:.+]] = arith.index_cast %arg1 : index to i32
// CHECK: call @polygeist_cusparse_spmv_jds_f32_sized
// CHECK-SAME: (%[[ROWS]], %[[REPETITIONS]],
// CHECK-NOT: kernel.launch
// CHECK: func.func private @polygeist_cusparse_spmv_jds_f32_sized
