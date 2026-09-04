// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s

module {
  // Parboil SpMV uses jagged diagonal storage: each row has a nonzero count,
  // and the column array supplies an indirect gather from x.
  func.func @jds_spmv(%counts: memref<?xi32>, %offsets: memref<?xi32>,
                      %columns: memref<?xi32>, %values: memref<?xf32>,
                      %x: memref<?xf32>, %y: memref<?xf32>, %rows: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %rows step %c1 {
      %count32 = memref.load %counts[%row] : memref<?xi32>
      %count = arith.index_cast %count32 : i32 to index
      %sum = scf.for %k = %c0 to %count step %c1
          iter_args(%acc = %zero) -> f32 {
        %base32 = memref.load %offsets[%k] : memref<?xi32>
        %row32 = arith.index_cast %row : index to i32
        %position32 = arith.addi %base32, %row32 : i32
        %position = arith.index_cast %position32 : i32 to index
        %column32 = memref.load %columns[%position] : memref<?xi32>
        %column = arith.index_cast %column32 : i32 to index
        %value = memref.load %values[%position] : memref<?xf32>
        %xvalue = memref.load %x[%column] : memref<?xf32>
        %product = arith.mulf %value, %xvalue : f32
        %next = arith.addf %acc, %product : f32
        scf.yield %next : f32
      }
      memref.store %sum, %y[%row] : memref<?xf32>
    }
    return
  }
}

// CHECK: residual_idiom_candidate body#[]
// CHECK-SAME: kind=jds_spmv
// CHECK-SAME: evidence=['row bounds loaded from %counts', 'column-indexed gather', 'multiply-add row reduction']
// CHECK-SAME: lowering_blocker=needs sparse operand-role validation and cuSPARSE ABI lowering
