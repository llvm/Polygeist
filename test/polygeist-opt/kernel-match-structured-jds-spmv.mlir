// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite | FileCheck %s --check-prefix=LIBRARY

module {
  // Parboil SpMV uses jagged diagonal storage: each row has a nonzero count,
  // and the column array supplies an indirect gather from x.
  func.func @jds_spmv(%counts: memref<?xi32>, %offsets: memref<?xi32>,
                      %columns: memref<?xi32>, %values: memref<?xf32>,
                      %permutation: memref<?xi32>, %x: memref<?xf32>,
                      %y: memref<?xf32>, %rows: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    affine.for %repeat = 0 to 50 {
      scf.for %row = %c0 to %rows step %c1 {
        %row32 = arith.index_cast %row : index to i32
        %count32 = memref.load %counts[%row] : memref<?xi32>
        %count = arith.index_cast %count32 : i32 to index
        %sum = scf.for %k = %c0 to %count step %c1
            iter_args(%acc = %zero) -> f32 {
          %base32 = memref.load %offsets[%k] : memref<?xi32>
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
        %permuted32 = memref.load %permutation[%row] : memref<?xi32>
        %permuted = arith.index_cast %permuted32 : i32 to index
        memref.store %sum, %y[%permuted] : memref<?xf32>
      }
    }
    return
  }
}

// CHECK: residual_idiom_candidate body#[]
// CHECK-SAME: kind=jds_spmv
// CHECK-SAME: evidence=['row bounds loaded from %counts', 'column-indexed gather', 'multiply-add row reduction']
// CHECK-SAME: lowering_blocker=cuSPARSE route available through the validated JDS-to-CSR storage adapter after exact operand-role validation

// LIBRARY-LABEL: func.func @jds_spmv
// LIBRARY: %[[REPEATS:.+]] = arith.constant 50 : index
// LIBRARY: kernel.launch @cusparseSpMV_JDS_f32_memref
// LIBRARY-SAME: (%rows, %[[REPEATS]], %counts, %offsets, %columns, %values, %permutation, %x, %y)
// LIBRARY-NOT: scf.{{for}}
