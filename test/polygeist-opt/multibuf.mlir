// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s --allow-unregistered-dialect | FileCheck %s

module {
  func.func @multi(%arg0: i32, %arg1: memref<?xmemref<?xi8>>, %arg2: index, %arg3: index) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c2_i32 = arith.constant 2 : i32
    %alloca = memref.alloca(%arg3) : memref<?x2xi32>
    scf.for %arg4 = %c0 to %arg3 step %c1 {
      %a = arith.index_cast %arg4 : index to i32
      memref.store %c2_i32, %alloca[%arg4, %c0] : memref<?x2xi32>
      memref.store %a, %alloca[%arg4, %c1] : memref<?x2xi32>
    }
    %a10 = memref.load %alloca[%arg2, %c0] : memref<?x2xi32>
    %a11 = memref.load %alloca[%arg2, %c1] : memref<?x2xi32>
    return %a10, %a11 : i32, i32
  }
}

// CHECK:   func.func @multi(%arg0: i32, %arg1: memref<?xmemref<?xi8>>, %arg2: index, %arg3: index) 
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %[[i0:.+]] = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:     return %c2_i32, %[[i0]] : i32, i32
// CHECK-NEXT:   }
