// RUN: polymer-opt %s -fold-scf-if -reg2mem -extract-scop-stmt | FileCheck %s
 
func @encrypt(%Sbox: memref<?x16xi32>, %statemt: memref<?xi32>)  {
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c15_i32 = arith.constant 15 : i32
  %c8_i32 = arith.constant 8 : i32
  %c283_i32 = arith.constant 283 : i32
  %ret = memref.alloca() : memref<1024xi32>
  affine.for %arg2 = 1 to 5 {
    affine.for %arg3 = 0 to 16 {
      %1 = affine.load %statemt[%arg3 * 4] : memref<?xi32>
      %2 = arith.shrsi %1, %c4_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.andi %1, %c15_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %Sbox[%3, %5] : memref<?x16xi32>
      affine.store %6, %statemt[%arg3 * 4] : memref<?xi32>
    }
    affine.for %arg3 = 0 to 1023 {
      %1 = affine.load %statemt[%arg3] : memref<?xi32>
      %2 = arith.shli %1, %c1_i32 : i32
      affine.store %2, %ret[%arg3] : memref<1024xi32>
      %3 = arith.shrsi %2, %c8_i32 : i32
      %4 = arith.cmpi eq, %3, %c1_i32 : i32
      scf.if %4 {
        %10 = arith.xori %2, %c283_i32 : i32
        affine.store %10, %ret[%arg3] : memref<1024xi32>
      }
      %5 = affine.load %statemt[%arg3 + 1] : memref<?xi32>
      %6 = arith.shli %5, %c1_i32 : i32
      %7 = arith.xori %5, %6 : i32
      %8 = arith.shrsi %7, %c8_i32 : i32
      %9 = arith.cmpi eq, %8, %c1_i32 : i32
      scf.if %9 {
        %10 = arith.xori %7, %c283_i32 : i32
        %11 = affine.load %ret[%arg3] : memref<1024xi32>
        %12 = arith.xori %11, %10 : i32
        affine.store %12, %ret[%arg3] : memref<1024xi32>
      } else {
        %10 = affine.load %ret[%arg3] : memref<1024xi32>
        %11 = arith.xori %10, %7 : i32
        affine.store %11, %ret[%arg3] : memref<1024xi32>
      }
    }
    affine.for %arg3 = 0 to 1024 {
      %1 = affine.load %ret[%arg3] : memref<1024xi32>
      affine.store %1, %statemt[%arg3] : memref<?xi32>
    }
  }
  return
}

// CHECK: func @encrypt(%[[Sbox:.*]]: memref<?x16xi32>, %[[statemt:.*]]: memref<?xi32>) 
// CHECK:   %[[v0:.*]] = memref.alloca() : memref<1024xi32>
// CHECK:   affine.for %[[i:.*]] = 1 to 5 
// CHECK:     affine.for %[[j:.*]] = 0 to 16 
// CHECK:       call @S0(%[[statemt]], %[[j]], %[[Sbox]])
// CHECK:     affine.for %[[j:.*]] = 0 to 1023 
// CHECK:       call @S1(%[[j]], %[[v0]], %[[statemt]])
// CHECK:       call @S2(%[[j]], %[[v0]], %[[statemt]])
// CHECK:     affine.for %[[j:.*]] = 0 to 1024 
// CHECK:       call @S3(%[[statemt]], %[[j]], %[[v0]])
