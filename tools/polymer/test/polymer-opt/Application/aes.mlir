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

// CHECK:  func private @S0(%[[statemt:.*]]: memref<?xi32>, %[[i:.*]]: index, %[[Sbox:.*]]: memref<?x16xi32>) attributes {scop.stmt} 
// CHECK-NEXT:    %[[c15_i32:.*]] = arith.constant 15 : i32
// CHECK-NEXT:    %[[c4_i32:.*]] = arith.constant 4 : i32
// CHECK-NEXT:    %[[v0:.*]] = affine.load %[[statemt]][symbol(%[[i]]) * 4] : memref<?xi32>
// CHECK-NEXT:    %[[v1:.*]] = arith.shrsi %[[v0]], %[[c4_i32]] : i32
// CHECK-NEXT:    %[[v2:.*]] = arith.index_cast %[[v1]] : i32 to index
// CHECK-NEXT:    %[[v3:.*]] = arith.andi %[[v0]], %[[c15_i32]] : i32
// CHECK-NEXT:    %[[v4:.*]] = arith.index_cast %[[v3]] : i32 to index
// CHECK-NEXT:    %[[v5:.*]] = memref.load %[[Sbox]][%[[v2]], %[[v4]]] : memref<?x16xi32>
// CHECK-NEXT:    affine.store %[[v5]], %[[statemt]][symbol(%[[i]]) * 4] : memref<?xi32>

// CHECK: func private @S1(%[[i:.*]]: index, %[[ret:.*]]: memref<1024xi32>, %[[statemt:.*]]: memref<?xi32>) attributes {scop.stmt} 
// CHECK-NEXT:   %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[v0:.*]] = affine.load %[[statemt]][symbol(%[[i]])] : memref<?xi32>
// CHECK-NEXT:   %[[v1:.*]] = arith.shli %[[v0]], %[[c1_i32]] : i32
// CHECK-NEXT:   affine.store %[[v1]], %[[ret]][symbol(%[[i]])] : memref<1024xi32>

// CHECK: func private @S2(%[[i:.*]]: index, %[[ret:.*]]: memref<1024xi32>) attributes {scop.stmt} 
// CHECK:   %[[c283_i32:.*]] = arith.constant 283 : i32
// CHECK:   %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK:   %[[c8_i32:.*]] = arith.constant 8 : i32
// CHECK:   %[[v0:.*]] = affine.load %[[ret]][symbol(%[[i]])] : memref<1024xi32>
// CHECK:   %[[v1:.*]] = arith.shrsi %[[v0]], %[[c8_i32]] : i32
// CHECK:   %[[v2:.*]] = arith.cmpi eq, %[[v1]], %[[c1_i32]] : i32
// CHECK:   %[[v3:.*]] = arith.xori %[[v0]], %[[c283_i32]] : i32
// CHECK:   %[[v4:.*]] = affine.load %[[ret]][symbol(%[[i]])] : memref<1024xi32>
// CHECK:   %[[v5:.*]] = select %[[v2]], %[[v3]], %[[v4]] : i32
// CHECK:   affine.store %[[v5]], %[[ret]][symbol(%[[i]])] : memref<1024xi32>

// CHECK: func private @S3(%[[i:.*]]: index, %[[ret:.*]]: memref<1024xi32>, %[[statemt]]: memref<?xi32>) attributes {scop.stmt} 
// CHECK-NEXT:   %[[c283_i32:.*]] = arith.constant 283 : i32
// CHECK-NEXT:   %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[c8_i32:.*]] = arith.constant 8 : i32
// CHECK-NEXT:   %[[v0:.*]] = affine.load %[[ret]][symbol(%[[i]])] : memref<1024xi32>
// CHECK-NEXT:   %[[v1:.*]] = affine.load %[[ret]][symbol(%[[i]])] : memref<1024xi32>
// CHECK-NEXT:   %[[v2:.*]] = affine.load %[[statemt]][symbol(%[[i]]) + 1] : memref<?xi32>
// CHECK-NEXT:   %[[v3:.*]] = arith.shli %[[v2]], %[[c1_i32]] : i32
// CHECK-NEXT:   %[[v4:.*]] = arith.xori %[[v2]], %[[v3]] : i32
// CHECK-NEXT:   %[[v5:.*]] = arith.shrsi %[[v4]], %[[c8_i32]] : i32
// CHECK-NEXT:   %[[v6:.*]] = arith.cmpi eq, %[[v5]], %[[c1_i32]] : i32
// CHECK-NEXT:   %[[v7:.*]] = arith.xori %[[v4]], %[[c283_i32]] : i32
// CHECK-NEXT:   %[[v8:.*]] = arith.xori %[[v0]], %[[v7]] : i32
// CHECK-NEXT:   %[[v9:.*]] = arith.xori %[[v1]], %[[v4]] : i32
// CHECK-NEXT:   %[[v10:.*]] = select %[[v6]], %[[v8]], %[[v9]] : i32
// CHECK-NEXT:   affine.store %[[v10]], %[[ret]][symbol(%[[i]])] : memref<1024xi32>

// CHECK: func private @S4(%[[statemt:.*]]: memref<?xi32>, %[[i:.*]]: index, %[[ret:.*]]: memref<1024xi32>) attributes {scop.stmt} 
// CHECK-NEXT:   %[[v0:.*]] = affine.load %[[ret]][symbol(%[[i]])] : memref<1024xi32>
// CHECK-NEXT:   affine.store %[[v0]], %[[statemt]][symbol(%[[i]])] : memref<?xi32>

// CHECK:  func @encrypt(%[[Sbox:.*]]: memref<?x16xi32>, %[[statemt:.*]]: memref<?xi32>)
// CHECK:    %[[ret:.*]] = memref.alloca() : memref<1024xi32>
// CHECK:    affine.for %[[i:.*]] = 1 to 5 
// CHECK:      affine.for %[[j:.*]] = 0 to 16 
// CHECK:        call @S0(%[[statemt]], %[[j]], %[[Sbox]]) : (memref<?xi32>, index, memref<?x16xi32>) -> ()
// CHECK:      affine.for %[[j:.*]] = 0 to 1023 
// CHECK:        call @S1(%[[j]], %[[ret]], %[[statemt]]) : (index, memref<1024xi32>, memref<?xi32>) -> ()
// CHECK:        call @S2(%[[j]], %[[ret]]) : (index, memref<1024xi32>) -> ()
// CHECK:        call @S3(%[[j]], %[[ret]], %[[statemt]]) : (index, memref<1024xi32>, memref<?xi32>) -> ()
// CHECK:      affine.for %[[j:.*]] = 0 to 1024 
// CHECK:        call @S4(%[[statemt]], %[[j]], %[[ret]]) : (memref<?xi32>, index, memref<1024xi32>) -> ()
