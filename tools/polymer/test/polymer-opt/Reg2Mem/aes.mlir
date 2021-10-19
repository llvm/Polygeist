// RUN: exit 0
 
func @encrypt(%arg0: memref<?x16xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
  %c1_i32 = constant 1 : i32
  %c4_i32 = constant 4 : i32
  %c15_i32 = constant 15 : i32
  %c8_i32 = constant 8 : i32
  %c283_i32 = constant 283 : i32
  %0 = memref.alloca() : memref<1024xi32>
  affine.for %arg2 = 1 to 5 {
    affine.for %arg3 = 0 to 16 {
      %1 = affine.load %arg1[%arg3 * 4] : memref<?xi32>
      %2 = shift_right_signed %1, %c4_i32 : i32
      %3 = index_cast %2 : i32 to index
      %4 = and %1, %c15_i32 : i32
      %5 = index_cast %4 : i32 to index
      %6 = memref.load %arg0[%3, %5] : memref<?x16xi32>
      affine.store %6, %arg1[%arg3 * 4] : memref<?xi32>
    }
    affine.for %arg3 = 0 to 1023 {
      %1 = affine.load %arg1[%arg3] : memref<?xi32>
      %2 = shift_left %1, %c1_i32 : i32
      affine.store %2, %0[%arg3] : memref<1024xi32>
      %3 = shift_right_signed %2, %c8_i32 : i32
      %4 = cmpi eq, %3, %c1_i32 : i32
      scf.if %4 {
        %10 = xor %2, %c283_i32 : i32
        affine.store %10, %0[%arg3] : memref<1024xi32>
      }
      %5 = affine.load %arg1[%arg3 + 1] : memref<?xi32>
      %6 = shift_left %5, %c1_i32 : i32
      %7 = xor %5, %6 : i32
      %8 = shift_right_signed %7, %c8_i32 : i32
      %9 = cmpi eq, %8, %c1_i32 : i32
      scf.if %9 {
        %10 = xor %7, %c283_i32 : i32
        %11 = affine.load %0[%arg3] : memref<1024xi32>
        %12 = xor %11, %10 : i32
        affine.store %12, %0[%arg3] : memref<1024xi32>
      } else {
        %10 = affine.load %0[%arg3] : memref<1024xi32>
        %11 = xor %10, %7 : i32
        affine.store %11, %0[%arg3] : memref<1024xi32>
      }
    }
    affine.for %arg3 = 0 to 1024 {
      %1 = affine.load %0[%arg3] : memref<1024xi32>
      affine.store %1, %arg1[%arg3] : memref<?xi32>
    }
  }
  return
}
