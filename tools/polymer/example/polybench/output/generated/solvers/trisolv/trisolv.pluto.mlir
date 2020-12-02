#map0 = affine_map<() -> (0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0)>
#map3 = affine_map<()[s0, s1] -> (s0, s1)>
#map4 = affine_map<()[s0] -> (s0, s0)>
#map5 = affine_map<() -> (1)>


module {
  func @kernel_trisolv(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %0 {
      call @S0(%arg2, %arg4, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      %1 = alloca() : memref<1xf64>
      call @S1(%1, %arg2, %arg4) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg5 = 0 to #map1(%arg4) {
        call @S2(%arg2, %arg4, %arg5, %arg1, %1) : (memref<2000xf64>, index, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
      call @S3(%arg2, %arg4, %arg1) : (memref<2000xf64>, index, memref<2000x2000xf64>) -> ()
    }
    return
  }
  func @S0(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[symbol(%arg1)] : memref<2000xf64>
    affine.store %0, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<2000xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<2000xf64>, %arg1: index, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<2000x2000xf64>
    %2 = affine.load %arg0[symbol(%arg2)] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @S3(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000x2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<2000xf64>
    %1 = affine.load %arg2[symbol(%arg1), symbol(%arg1)] : memref<2000x2000xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @kernel_trisolv_new(%arg0: memref<2000xf64>, %arg1: memref<2000xf64>, %arg2: memref<2000x2000xf64>, %arg3: i32) {
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg3 : i32 to index
    call @S0(%arg1, %c0, %arg0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
    call @S1(%0, %arg1, %c0) : (memref<1xf64>, memref<2000xf64>, index) -> ()
    call @S3(%arg1, %c0, %arg2) : (memref<2000xf64>, index, memref<2000x2000xf64>) -> ()
    affine.for %arg4 = 1 to %1 {
      call @S0(%arg1, %arg4, %arg0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      call @S1(%0, %arg1, %arg4) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg5 = 0 to #map1(%arg4) {
        call @S2(%arg1, %arg4, %arg5, %arg2, %0) : (memref<2000xf64>, index, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
      call @S3(%arg1, %arg4, %arg2) : (memref<2000xf64>, index, memref<2000x2000xf64>) -> ()
    }
    return
  }
}
