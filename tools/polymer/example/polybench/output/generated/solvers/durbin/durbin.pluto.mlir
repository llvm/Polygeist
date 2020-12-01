#map0 = affine_map<() -> (0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (1)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0) -> (1, d0 * 32)>
#map6 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map7 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>


module {
  func @kernel_durbin(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
    %0 = alloca() : memref<2000xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    call @S0(%arg2, %arg1) : (memref<2000xf64>, memref<2000xf64>) -> ()
    call @S1(%2) : (memref<1xf64>) -> ()
    call @S2(%1, %arg1) : (memref<1xf64>, memref<2000xf64>) -> ()
    %4 = index_cast %arg0 : i32 to index
    %5 = alloca() : memref<1xf64>
    call @S3(%5, %2, %1) : (memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
    call @S4(%2, %1) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S5(%3) : (memref<1xf64>) -> ()
    %6 = alloca() : memref<1xf64>
    call @S6(%6, %3) : (memref<1xf64>, memref<1xf64>) -> ()
    %7 = alloca() : memref<1xf64>
    call @S7(%7, %3) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg3 = 1 to %4 {
      affine.for %arg4 = 0 to #map1(%arg3) {
        call @S8(%3, %arg2, %arg4, %arg1, %6) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, memref<1xf64>) -> ()
      }
      %8 = alloca() : memref<1xf64>
      call @S9(%8, %5, %7, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S10(%1, %5, %7, %arg1, %arg3) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg4 = 0 to #map1(%arg3) {
        call @S11(%arg4, %0, %arg2, %8) : (index, memref<2000xf64>, memref<2000xf64>, memref<1xf64>) -> ()
      }
      affine.for %arg4 = 0 to #map1(%arg3) {
        call @S12(%arg2, %arg4, %0) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
      call @S13(%arg2, %arg3, %5, %7, %arg1) : (memref<2000xf64>, index, memref<1xf64>, memref<1xf64>, memref<2000xf64>) -> ()
    }
    return
  }
  func @S0(%arg0: memref<2000xf64>, %arg1: memref<2000xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<2000xf64>
    %1 = negf %0 : f64
    affine.store %1, %arg0[%c0] : memref<2000xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: memref<2000xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<2000xf64>
    %1 = negf %0 : f64
    affine.store %1, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1xf64>, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c1_i32 : i32 to f64
    %c0 = constant 0 : index
    %1 = affine.load %arg2[%c0] : memref<1xf64>
    %2 = affine.load %arg2[%c0] : memref<1xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %5 = affine.load %arg1[%c0] : memref<1xf64>
    %6 = mulf %4, %5 : f64
    affine.store %6, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c1_i32 : i32 to f64
    %c0 = constant 0 : index
    %1 = affine.load %arg1[%c0] : memref<1xf64>
    %2 = affine.load %arg1[%c0] : memref<1xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %5 = affine.load %arg0[%c0] : memref<1xf64>
    %6 = mulf %4, %5 : f64
    affine.store %6, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    %c0 = constant 0 : index
    affine.store %cst, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S6(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S7(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S8(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000xf64>, %arg4: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[0] : memref<1xf64>
    %1 = affine.apply #map4(%arg2)
    %2 = affine.load %arg3[%1] : memref<2000xf64>
    %3 = affine.load %arg1[%arg2] : memref<2000xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    %c0 = constant 0 : index
    affine.store %5, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S9(%arg0: memref<1xf64>, %arg1: memref<1xf64>, %arg2: memref<1xf64>, %arg3: memref<2000xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg3[%arg4] : memref<2000xf64>
    %1 = affine.load %arg2[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = affine.load %arg1[0] : memref<1xf64>
    %5 = divf %3, %4 : f64
    affine.store %5, %arg0[0] : memref<1xf64>
    return
  }
  func @S10(%arg0: memref<1xf64>, %arg1: memref<1xf64>, %arg2: memref<1xf64>, %arg3: memref<2000xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg3[%arg4] : memref<2000xf64>
    %1 = affine.load %arg2[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = affine.load %arg1[0] : memref<1xf64>
    %5 = divf %3, %4 : f64
    %c0 = constant 0 : index
    affine.store %5, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S11(%arg0: index, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg0] : memref<2000xf64>
    %1 = affine.load %arg3[0] : memref<1xf64>
    %2 = affine.apply #map4(%arg0)
    %3 = affine.load %arg2[%2] : memref<2000xf64>
    %4 = mulf %1, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg1[%arg0] : memref<2000xf64>
    return
  }
  func @S12(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[%arg1] : memref<2000xf64>
    affine.store %0, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @S13(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<1xf64>, %arg3: memref<1xf64>, %arg4: memref<2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[%arg1] : memref<2000xf64>
    %1 = affine.load %arg3[0] : memref<1xf64>
    %2 = addf %0, %1 : f64
    %3 = negf %2 : f64
    %4 = affine.load %arg2[0] : memref<1xf64>
    %5 = divf %3, %4 : f64
    affine.store %5, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @kernel_durbin_new(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = alloca() : memref<1xf64>
    %5 = alloca() : memref<1xf64>
    %6 = index_cast %arg0 : i32 to index
    %7 = alloca() : memref<2000xf64>
    call @S5(%2) : (memref<1xf64>) -> ()
    call @S7(%0, %2) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S6(%1, %2) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S2(%4, %arg1) : (memref<1xf64>, memref<2000xf64>) -> ()
    call @S1(%5) : (memref<1xf64>) -> ()
    call @S3(%3, %5, %4) : (memref<1xf64>, memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg3 = 0 to #map7()[%6] {
      %8 = alloca() : memref<1xf64>
      affine.for %arg4 = max #map5(%arg3) to min #map6(%arg3)[%6] {
        %9 = alloca() : memref<1xf64>
        %10 = alloca() : memref<1xf64>
        %11 = affine.apply #map1(%arg4)
        call @S13(%arg2, %11, %10, %9, %arg1) : (memref<2000xf64>, index, memref<1xf64>, memref<1xf64>, memref<2000xf64>) -> ()
      }
    }
    call @S4(%5, %4) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg3 = 1 to %6 {
      %8 = alloca() : memref<1xf64>
      %9 = alloca() : memref<1xf64>
      %10 = alloca() : memref<1xf64>
      %11 = affine.apply #map1(%arg3)
      call @S10(%10, %9, %8, %arg1, %11) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
    }
    call @S0(%arg2, %arg1) : (memref<2000xf64>, memref<2000xf64>) -> ()
    affine.for %arg3 = 1 to %6 {
      %8 = alloca() : memref<2000xf64>
      %9 = alloca() : memref<1xf64>
      %10 = alloca() : memref<1xf64>
      %11 = alloca() : memref<2000xf64>
      %12 = alloca() : memref<1xf64>
      %13 = alloca() : memref<1xf64>
      %14 = alloca() : memref<1xf64>
      %15 = affine.apply #map1(%arg3)
      call @S9(%14, %13, %12, %arg1, %15) : (memref<1xf64>, memref<1xf64>, memref<1xf64>, memref<2000xf64>, index) -> ()
      call @S11(%15, %11, %arg2, %14) : (index, memref<2000xf64>, memref<2000xf64>, memref<1xf64>) -> ()
      call @S8(%10, %arg2, %15, %arg1, %9) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, memref<1xf64>) -> ()
      affine.for %arg4 = 1 to #map1(%arg3) {
        %16 = alloca() : memref<1xf64>
        %17 = alloca() : memref<1xf64>
        %18 = alloca() : memref<1xf64>
        %19 = alloca() : memref<2000xf64>
        call @S11(%15, %19, %arg2, %18) : (index, memref<2000xf64>, memref<2000xf64>, memref<1xf64>) -> ()
        call @S12(%arg2, %15, %19) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        call @S8(%17, %arg2, %15, %arg1, %16) : (memref<1xf64>, memref<2000xf64>, index, memref<2000xf64>, memref<1xf64>) -> ()
      }
      call @S12(%arg2, %15, %8) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
    }
    return
  }
}
