// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

module  {
  func.func private @run() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca() : memref<40xi32>
    %tmp = memref.alloc() : memref<40xi32>
    affine.for %arg4 = 0 to 40 {
        %v = affine.load %prev[%arg4] : memref<40xi32>
        affine.store %v, %tmp[%arg4] : memref<40xi32>
    }
    affine.for %arg4 = 0 to 22 {
        %v = affine.load %tmp[%arg4] : memref<40xi32>
        %v2 = arith.addi %v, %v : i32
        memref.store %v2, %tmp[%arg4] : memref<40xi32>
    }
    affine.for %arg4 = 0 to 40 {
        %v = affine.load %tmp[%arg4] : memref<40xi32>
        affine.store %v, %prev[%arg4] : memref<40xi32>
    }
    memref.dealloc %tmp : memref<40xi32>
    return
  }
  func.func private @runMM() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca() : memref<40xi32>
    %tmp = memref.alloc() : memref<40xi32>
    affine.for %arg4 = 0 to 40 {
        %v = affine.load %prev[%arg4] : memref<40xi32>
        affine.store %v, %tmp[%arg4] : memref<40xi32>
    }
    affine.for %arg4 = 0 to 22 {
        %v = affine.load %tmp[%arg4] : memref<40xi32>
        %v2 = arith.addi %v, %v : i32
        memref.store %v2, %tmp[%arg4] : memref<40xi32>
    }
    affine.for %arg4 = 0 to 20 {
        %v = affine.load %tmp[%arg4] : memref<40xi32>
        affine.store %v, %prev[%arg4] : memref<40xi32>
    }
    memref.dealloc %tmp : memref<40xi32>
    return
  }
  func.func private @runUS(%s : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%s) : memref<?xi32>
    %tmp = memref.alloc(%s) : memref<?xi32>
    affine.for %arg4 = 0 to %s {
        %v = affine.load %prev[%arg4] : memref<?xi32>
        affine.store %v, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to 22 {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        %v2 = arith.addi %v, %v : i32
        memref.store %v2, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to %s {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        affine.store %v, %prev[%arg4] : memref<?xi32>
    }
    memref.dealloc %tmp : memref<?xi32>
    return
  }
  func.func private @runUSMM(%s : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%s) : memref<?xi32>
    %tmp = memref.alloc(%s) : memref<?xi32>
    affine.for %arg4 = 0 to 40 {
        %v = affine.load %prev[%arg4] : memref<?xi32>
        affine.store %v, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to 22 {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        %v2 = arith.addi %v, %v : i32
        memref.store %v2, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to %s {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        affine.store %v, %prev[%arg4] : memref<?xi32>
    }
    memref.dealloc %tmp : memref<?xi32>
    return
  }
  func.func private @runDouble(%s : index, %d: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%s) : memref<?xi32>
    %tmp = memref.alloc(%s) : memref<?xi32>
    affine.for %arg4 = 0 to %s {
        %v = affine.load %d[%arg4] : memref<?xi32>
        affine.store %v, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to %s {
        %v = affine.load %prev[%arg4] : memref<?xi32>
        affine.store %v, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to 22 {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        %v2 = arith.addi %v, %v : i32
        memref.store %v2, %tmp[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to %s {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        affine.store %v, %prev[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to %s {
        %v = affine.load %tmp[%arg4] : memref<?xi32>
        affine.store %v, %d[%arg4] : memref<?xi32>
    }
    memref.dealloc %tmp : memref<?xi32>
    return
  }
}

// CHECK:   func.func private @run() {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<40xi32>
// CHECK-NEXT:     affine.for %[[arg0:.+]] = 0 to 22 {
// CHECK-NEXT:       %[[V1:.+]] = affine.load %[[V0]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:       %[[V2:.+]] = arith.addi %[[V1]], %[[V1]] : i32
// CHECK-NEXT:       memref.store %[[V2]], %[[V0]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func private @runMM() {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<40xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloc() : memref<40xi32>
// CHECK-NEXT:     affine.for %[[arg0:.+]] = 0 to 40 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V0]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V1]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg0:.+]] = 0 to 22 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V1]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:       %[[V3:.+]] = arith.addi %[[V2]], %[[V2]] : i32
// CHECK-NEXT:       memref.store %[[V3]], %[[V1]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg0:.+]] = 0 to 20 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V1]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V0]][%[[arg0]]] : memref<40xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %[[V1]] : memref<40xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// CHECK:   func.func private @runUS(%[[arg0:.+]]: index)
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca(%[[arg0]]) : memref<?xi32>
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 22 {
// CHECK-NEXT:       %[[V1:.+]] = affine.load %[[V0]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:       %[[V2:.+]] = arith.addi %[[V1]], %[[V1]] : i32
// CHECK-NEXT:       memref.store %[[V2]], %[[V0]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func private @runUSMM(%[[arg0:.+]]: index) {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca(%[[arg0]]) : memref<?xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloc(%[[arg0]]) : memref<?xi32>
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 40 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V0]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V1]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 22 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V1]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:       %[[V3:.+]] = arith.addi %[[V2]], %[[V2]] : i32
// CHECK-NEXT:       memref.store %[[V3]], %[[V1]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to %[[arg0]] {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V1]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V0]][%[[arg1]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %[[V1]] : memref<?xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func private @runDouble(%[[arg0:.+]]: index, %[[arg1:.+]]: memref<?xi32>) {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca(%[[arg0]]) : memref<?xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloc(%[[arg0]]) : memref<?xi32>
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to %[[arg0]] {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[arg1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to 22 {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:       %[[V3:.+]] = arith.addi %[[V2]], %[[V2]] : i32
// CHECK-NEXT:       memref.store %[[V3]], %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to %[[arg0]] {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[V1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to %[[arg0]] {
// CHECK-NEXT:       %[[V2:.+]] = affine.load %[[V1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[V2]], %[[arg1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %[[V1]] : memref<?xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
