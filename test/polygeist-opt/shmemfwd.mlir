// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s --allow-unregistered-dialect | FileCheck %s

module {
  func.func private @print1(f32)
  func.func @main() {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca() : memref<2xf32>
    scf.parallel (%arg2) = (%c0) to (%c2) step (%c1) {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.sitofp %0 : i32 to f32
      memref.store %1, %alloca[%arg2] : memref<2xf32>
      "polygeist.barrier"(%arg2, %c0) : (index, index) -> ()
      %2 = arith.cmpi eq, %arg2, %c1 : index
      scf.if %2 {
        %3 = arith.addi %arg2, %c-1 : index
        %4 = memref.load %alloca[%3] : memref<2xf32>
        func.call @print1(%4) : (f32) -> ()
      }
      scf.yield
    }
    return
  }
}

// CHECK:   func.func @main() {
// CHECK-NEXT:     %[[cm1:.+]] = arith.constant -1 : index
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[c2:.+]] = arith.constant 2 : index
// CHECK-NEXT:     scf.parallel (%[[arg0:.+]]) = (%[[c0]]) to (%[[c2]]) step (%[[c1]]) {
// CHECK-NEXT:       %[[i0:.+]] = arith.cmpi eq, %arg0, %[[c1]] : index
// CHECK-NEXT:       scf.if %[[i0]] {
// CHECK-NEXT:         %[[i1:.+]] = arith.addi %[[arg0]], %[[cm1]] : index
// CHECK-NEXT:         %[[i2:.+]] = arith.index_cast %[[i1]] : index to i32
// CHECK-NEXT:         %[[i3:.+]] = arith.sitofp %[[i2]] : i32 to f32
// CHECK-NEXT:         func.call @print1(%[[i3]]) : (f32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
