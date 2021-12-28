// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func @_Z9calc_pathi(%arg0: i32, %c : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %c9 = arith.constant 9 : index
    %true = arith.constant true
      %23 = memref.alloca() : memref<256xi32>
      scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
          %26 = scf.if %c -> i1 {
            memref.store %c0_i32, %23[%c0] : memref<256xi32>
            "polygeist.barrier"() : () -> ()
            scf.yield %true : i1
          } else {
            scf.yield %false : i1
          }
          scf.yield
      }
    return
  }
}

// CHECK:   func @_Z9calc_pathi(%arg0: i32, %arg1: i1) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c9 = arith.constant 9 : index
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = memref.alloca() : memref<256xi32>
// CHECK-NEXT:     %1 = memref.alloc(%c9) : memref<?xmemref<1xi1>>
// CHECK-NEXT:     %2 = memref.alloc(%c9) : memref<?xindex>
// CHECK-NEXT:     scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:       %4 = llvm.mlir.undef : i1
// CHECK-NEXT:       %5 = arith.extui %arg1 : i1 to i64
// CHECK-NEXT:       %6 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:       memref.store %6, %2[%arg2] : memref<?xindex>
// CHECK-NEXT:       %7 = memref.alloc() : memref<1xi1>
// CHECK-NEXT:       memref.store %7, %1[%arg2] : memref<?xmemref<1xi1>>
// CHECK-NEXT:       memref.store %4, %7[%c0] : memref<1xi1>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = memref.load %2[%c0] : memref<?xindex>
// CHECK-NEXT:     scf.for %arg2 = %c0 to %3 step %c1 {
// CHECK-NEXT:       scf.parallel (%arg3) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:         memref.store %c0_i32, %0[%c0] : memref<256xi32>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.parallel (%arg3) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:         %4 = memref.load %1[%arg3] : memref<?xmemref<1xi1>>
// CHECK-NEXT:         memref.store %true, %4[%c0] : memref<1xi1>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:       %4 = memref.load %1[%arg2] : memref<?xmemref<1xi1>>
// CHECK-NEXT:       memref.dealloc %4 : memref<1xi1>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %1 : memref<?xmemref<1xi1>>
// CHECK-NEXT:     memref.dealloc %2 : memref<?xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
