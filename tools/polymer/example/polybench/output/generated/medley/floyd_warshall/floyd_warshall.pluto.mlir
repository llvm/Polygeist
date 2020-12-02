module  {
  func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<2800x2800xi32>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to %0 {
        %1 = alloca() : memref<1xi32>
        call @S0(%1, %arg1, %arg3, %arg2) : (memref<1xi32>, memref<2800x2800xi32>, index, index) -> ()
        affine.for %arg4 = 0 to %0 {
          call @S1(%arg1, %arg3, %arg4, %arg2, %1) : (memref<2800x2800xi32>, index, index, index, memref<1xi32>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1xi32>, %arg1: memref<2800x2800xi32>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2), symbol(%arg3)] : memref<2800x2800xi32>
    affine.store %0, %arg0[0] : memref<1xi32>
    return
  }
  func @S1(%arg0: memref<2800x2800xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1xi32>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<2800x2800xi32>
    %1 = affine.load %arg4[0] : memref<1xi32>
    %2 = affine.load %arg0[symbol(%arg3), symbol(%arg2)] : memref<2800x2800xi32>
    %3 = addi %1, %2 : i32
    %4 = cmpi "slt", %0, %3 : i32
    %5 = scf.if %4 -> (i32) {
      %6 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<2800x2800xi32>
      scf.yield %6 : i32
    } else {
      %6 = affine.load %arg0[symbol(%arg1), symbol(%arg3)] : memref<2800x2800xi32>
      %7 = affine.load %arg0[symbol(%arg3), symbol(%arg2)] : memref<2800x2800xi32>
      %8 = addi %6, %7 : i32
      scf.yield %8 : i32
    }
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<2800x2800xi32>
    return
  }
  func @kernel_floyd_warshall_new(%arg0: i32, %arg1: memref<2800x2800xi32>) {
    %0 = alloca() : memref<1xi32>
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %1 {
      affine.for %arg3 = 0 to %1 {
        call @S0(%0, %arg1, %arg2, %arg3) : (memref<1xi32>, memref<2800x2800xi32>, index, index) -> ()
        affine.for %arg4 = 0 to %1 {
          call @S1(%arg1, %arg2, %arg3, %arg4, %0) : (memref<2800x2800xi32>, index, index, index, memref<1xi32>) -> ()
        }
      }
    }
    return
  }
}

