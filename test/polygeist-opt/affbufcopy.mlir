// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s -allow-unregistered-dialect | FileCheck %s

module  {
  func.func private @print3(i32, i32, i32) -> ()
  func.func private @print1(i32) -> ()
  func.func private @run(%c: i32, %rng : index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run(%arg0: i32, %arg1: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     affine.parallel (%arg2, %arg3) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i1:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 4] : memref<?xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 11] : memref<?xi32>
// CHECK-NEXT:       func.call @print3(%[[i1]], %[[i2]], %[[i3]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }
  
  func.func private @run2(%c: i32, %rng : index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    %v = affine.load %tmp[3, 4] : memref<16x16xi32>
    func.call @print1(%v) : (i32) -> ()
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run2(%arg0: i32, %arg1: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     %[[i1:.+]] = memref.alloc() : memref<16x16xi32>
// CHECK-NEXT:     affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0]][%arg2 * 3 + %arg3 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:       affine.store %[[i3]], %[[i1]][%arg3, %arg2] : memref<16x16xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg2, %arg3) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:       %[[i4:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 4] : memref<?xi32>
// CHECK-NEXT:       %[[i5:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 11] : memref<?xi32>
// CHECK-NEXT:       func.call @print3(%[[i3]], %[[i4]], %[[i5]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     %[[i2:.+]] = affine.load %[[i1]][3, 4] : memref<16x16xi32>
// CHECK-NEXT:     call @print1(%[[i2]]) : (i32) -> ()
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }

  func.func private @run3(%c: i32, %rng : index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
      }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run3(%arg0: i32, %arg1: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     affine.parallel (%arg2, %arg3) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i1:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 4] : memref<?xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0]][%arg3 * 3 + %arg2 * 10 + 11] : memref<?xi32>
// CHECK-NEXT:       func.call @print3(%[[i1]], %[[i2]], %[[i3]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }

  func.func private @run4(%c: i32, %rng : index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg4, %arg4] : memref<16x16xi32>
      }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run4(%arg0: i32, %arg1: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     %[[i1:.+]] = memref.alloc() : memref<16x16xi32>
// CHECK-NEXT:     affine.for %arg2 = 0 to 16 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 16 {
// CHECK-NEXT:         %[[i2:.+]] = affine.load %[[i0]][%arg2 * 3 + %arg3 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:         affine.store %[[i2]], %[[i1]][%arg2, %arg2] : memref<16x16xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg2, %arg3) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i1:.+]][%arg2, %arg3] : memref<16x16xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i1:.+]][%arg2, %arg3 + 1] : memref<16x16xi32>
// CHECK-NEXT:       %[[i4:.+]] = affine.load %[[i1:.+]][%arg2 + 1, %arg3] : memref<16x16xi32>
// CHECK-NEXT:       func.call @print3(%[[i2]], %[[i3]], %[[i4]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }
  
 func.func private @run5(%c: i32, %rng : index, %cmp: index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg0 = 0 to %cmp {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
      }
    }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run5(%arg0: i32, %arg1: index, %arg2: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     %[[i1:.+]] = memref.alloc() : memref<16x16xi32>
// CHECK-NEXT:     affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:       affine.for %arg4 = 0 to 16 {
// CHECK-NEXT:         affine.for %arg5 = 0 to 16 {
// CHECK-NEXT:           %[[i2:.+]] = affine.load %[[i0]][%arg4 * 3 + %arg5 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:           affine.store %[[i2]], %[[i1]][%arg5, %arg4] : memref<16x16xi32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg3, %arg4) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i1:.+]][%arg3, %arg4] : memref<16x16xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i1:.+]][%arg3, %arg4 + 1] : memref<16x16xi32>
// CHECK-NEXT:       %[[i4:.+]] = affine.load %[[i1:.+]][%arg3 + 1, %arg4] : memref<16x16xi32>
// CHECK-NEXT:       func.call @print3(%[[i2]], %[[i3]], %[[i4]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }
  
 func.func private @run6(%c: i32, %rng : index, %cmp: index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg0 = 0 to 2 {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[1 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
      }
    }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run6(%arg0: i32, %arg1: index, %arg2: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     affine.parallel (%arg3, %arg4) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i1:.+]] = affine.load %[[i0:.+]][%arg4 * 3 + %arg3 * 10 + 1] : memref<?xi32>
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i0:.+]][%arg4 * 3 + %arg3 * 10 + 4] : memref<?xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0:.+]][%arg4 * 3 + %arg3 * 10 + 11] : memref<?xi32>
// CHECK-NEXT:       func.call @print3(%[[i1]], %[[i2]], %[[i3]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }
  
 func.func private @run7(%c: i32, %rng : index, %cmp: index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg0 = 0 to 2 {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[%arg0 +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
      }
    }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run7(%arg0: i32, %arg1: index, %arg2: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     %[[i1:.+]] = memref.alloc() : memref<16x16xi32>
// CHECK-NEXT:     affine.for %arg3 = 0 to 2 {
// CHECK-NEXT:       affine.for %arg4 = 0 to 16 {
// CHECK-NEXT:         affine.for %arg5 = 0 to 16 {
// CHECK-NEXT:           %[[i2:.+]] = affine.load %[[i0]][%arg3 + %arg4 * 3 + %arg5 * 10] : memref<?xi32>
// CHECK-NEXT:           affine.store %[[i2]], %[[i1]][%arg5, %arg4] : memref<16x16xi32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg3, %arg4) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i1]][%arg3, %arg4] : memref<16x16xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i1]][%arg3, %arg4 + 1] : memref<16x16xi32>
// CHECK-NEXT:       %[[i4:.+]] = affine.load %[[i1]][%arg3 + 1, %arg4] : memref<16x16xi32>
// CHECK-NEXT:       call @print3(%[[i2]], %[[i3]], %[[i4]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }
 
 func.func private @run8(%c: i32, %rng : index, %cmp: index, %arg0: index) -> memref<?xi32> {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %prev = memref.alloca(%rng) : memref<?xi32>
    %tmp = memref.alloc() : memref<16x16xi32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %v = affine.load %prev[symbol(%arg0) +  %arg4 * 3 + %arg5 * 10] : memref<?xi32>
        affine.store %v, %tmp[%arg5, %arg4] : memref<16x16xi32>
      }
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
        %v = affine.load %tmp[%arg4, %arg5] : memref<16x16xi32>
        %v2 = affine.load %tmp[%arg4, 1 + %arg5] : memref<16x16xi32>
        %v3 = affine.load %tmp[1 + %arg4, %arg5] : memref<16x16xi32>
        func.call @print3(%v, %v2, %v3) : (i32, i32, i32) -> ()
    }
    affine.store %c, %prev[0] : memref<?xi32>
    return %prev : memref<?xi32>
  }

// CHECK:   func.func private @run8(%arg0: i32, %arg1: index, %arg2: index, %arg3: index) -> memref<?xi32> {
// CHECK-NEXT:     %[[i0:.+]] = memref.alloca(%arg1) : memref<?xi32>
// CHECK-NEXT:     affine.parallel (%arg4, %arg5) = (0, 0) to (10, 10) {
// CHECK-NEXT:       %[[i1:.+]] = affine.load %[[i0]][%arg5 * 3 + %arg4 * 10 + symbol(%arg3)] : memref<?xi32>
// CHECK-NEXT:       %[[i2:.+]] = affine.load %[[i0]][%arg5 * 3 + %arg4 * 10 + symbol(%arg3) + 3] : memref<?xi32>
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[i0]][%arg5 * 3 + %arg4 * 10 + symbol(%arg3) + 10] : memref<?xi32>
// CHECK-NEXT:       func.call @print3(%[[i1]], %[[i2]], %[[i3]]) : (i32, i32, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %arg0, %[[i0]][0] : memref<?xi32>
// CHECK-NEXT:     return %[[i0]] : memref<?xi32>
// CHECK-NEXT:   }

}
