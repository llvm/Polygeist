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
            "polygeist.barrier"(%arg4) : (index) -> ()
            scf.yield %true : i1
          } else {
            scf.yield %false : i1
          }
          scf.yield
      }
    return
  }
}

// CHECK:   func @_Z9calc_pathi(%arg0: i32, %arg1: i1)
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %c9 = arith.constant 9 : index
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = memref.alloca() : memref<256xi32>
// CHECK-NEXT:     %1 = memref.alloc(%c9) : memref<?xi1>
// CHECK-NEXT:     scf.if %arg1 {
// CHECK-NEXT:       scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:         memref.store %c0_i32, %0[%c0] : memref<256xi32>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:         %2 = "polygeist.subindex"(%1, %arg2) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:         memref.store %true, %2[] : memref<i1>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:         %2 = "polygeist.subindex"(%1, %arg2) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:         memref.store %false, %2[] : memref<i1>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %1 : memref<?xi1>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
