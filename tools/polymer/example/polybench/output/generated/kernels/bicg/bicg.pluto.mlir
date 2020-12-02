#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<()[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


module {
  func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<2100x1900xf64>, %arg3: memref<1900xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>, %arg6: memref<2100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = alloca() : memref<1xf64>
    call @S0(%1) : (memref<1xf64>) -> ()
    affine.for %arg7 = 0 to %0 {
      call @S1(%arg3, %arg7, %1) : (memref<1900xf64>, index, memref<1xf64>) -> ()
    }
    %2 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %2 {
      call @S2(%arg4, %arg7) : (memref<2100xf64>, index) -> ()
      %3 = alloca() : memref<1xf64>
      call @S3(%3, %arg6, %arg7) : (memref<1xf64>, memref<2100xf64>, index) -> ()
      %4 = alloca() : memref<1xf64>
      call @S4(%4, %arg4, %arg7) : (memref<1xf64>, memref<2100xf64>, index) -> ()
      affine.for %arg8 = 0 to %0 {
        call @S5(%arg3, %arg8, %arg2, %arg7, %3) : (memref<1900xf64>, index, memref<2100x1900xf64>, index, memref<1xf64>) -> ()
        call @S6(%arg4, %arg7, %arg5, %arg8, %arg2, %4) : (memref<2100xf64>, index, memref<1900xf64>, index, memref<2100x1900xf64>, memref<1xf64>) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %c0_i32 = constant 0 : i32
    %0 = sitofp %c0_i32 : i32 to f64
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1900xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    affine.store %0, %arg0[symbol(%arg1)] : memref<1900xf64>
    return
  }
  func @S2(%arg0: memref<2100xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1)] : memref<2100xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<2100xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<2100xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<2100xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<2100xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1900xf64>, %arg1: index, %arg2: memref<2100x1900xf64>, %arg3: index, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<1900xf64>
    %1 = affine.load %arg4[0] : memref<1xf64>
    %2 = affine.load %arg2[symbol(%arg3), symbol(%arg1)] : memref<2100x1900xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<1900xf64>
    return
  }
  func @S6(%arg0: memref<2100xf64>, %arg1: index, %arg2: memref<1900xf64>, %arg3: index, %arg4: memref<2100x1900xf64>, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg3)] : memref<2100x1900xf64>
    %2 = affine.load %arg2[symbol(%arg3)] : memref<1900xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<2100xf64>
    return
  }
  func @kernel_bicg_new(%arg0: memref<1900xf64>, %arg1: memref<2100xf64>, %arg2: memref<2100xf64>, %arg3: memref<2100x1900xf64>, %arg4: memref<1900xf64>, %arg5: i32, %arg6: i32) {
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg6 : i32 to index
    %2 = index_cast %arg5 : i32 to index
    call @S0(%0) : (memref<1xf64>) -> ()
    affine.for %arg7 = 0 to #map5()[%1] {
      affine.for %arg8 = #map3(%arg7) to min #map4(%arg7)[%1] {
        call @S1(%arg0, %arg8, %0) : (memref<1900xf64>, index, memref<1xf64>) -> ()
      }
    }
    affine.for %arg7 = 0 to %1 {
      %3 = alloca() : memref<1xf64>
      %4 = alloca() : memref<1xf64>
      call @S2(%arg1, %arg7) : (memref<2100xf64>, index) -> ()
      call @S3(%4, %arg2, %arg7) : (memref<1xf64>, memref<2100xf64>, index) -> ()
      call @S4(%3, %arg1, %arg7) : (memref<1xf64>, memref<2100xf64>, index) -> ()
      affine.for %arg8 = 0 to %2 {
        call @S5(%arg0, %arg7, %arg3, %arg8, %4) : (memref<1900xf64>, index, memref<2100x1900xf64>, index, memref<1xf64>) -> ()
        call @S6(%arg1, %arg7, %arg4, %arg8, %arg3, %3) : (memref<2100xf64>, index, memref<1900xf64>, index, memref<2100x1900xf64>, memref<1xf64>) -> ()
      }
    }
    return
  }
}
