#map0 = affine_map<() -> (0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0) -> (d0, d0)>
#map6 = affine_map<() -> (2)>
#map7 = affine_map<(d0) -> (d0 * 32)>
#map8 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map9 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map10 = affine_map<(d0) -> (d0 + 1)>

#set0 = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 - 1 >= 0)>

module {
  func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
    %c1 = constant 1 : index
    %0 = alloca() : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    %2 = alloca() : memref<1xf64>
    call @S0(%2, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %3 = alloca() : memref<1xf64>
    call @S1(%3, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %4 = alloca() : memref<1xf64>
    call @S2(%4, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %5 = alloca() : memref<1xf64>
    call @S3(%5, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to %1 {
      affine.for %arg6 = 0 to #map1(%arg5) {
        call @S4(%0, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map1(%arg6) {
          call @S5(%0, %arg1, %arg7, %arg6, %arg5, %2) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index, memref<1xf64>) -> ()
        }
        call @S6(%arg1, %arg5, %arg6, %3) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
      affine.for %arg6 = #map1(%arg5) to %1 {
        call @S7(%0, %arg1, %arg5, %arg6) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        affine.for %arg7 = 0 to #map1(%arg5) {
          call @S8(%0, %arg1, %arg7, %arg6, %arg5, %4) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index, memref<1xf64>) -> ()
        }
        call @S9(%arg1, %arg5, %arg6, %5) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      }
    }
    %6 = alloca() : memref<1xf64>
    call @S10(%6, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %7 = alloca() : memref<1xf64>
    call @S11(%7, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to %1 {
      call @S12(%0, %arg2, %arg5) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg6 = 0 to #map1(%arg5) {
        call @S13(%0, %arg4, %arg6, %arg1, %arg5, %6) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
      }
      call @S14(%arg4, %arg5, %7) : (memref<2000xf64>, index, memref<1xf64>) -> ()
    }
    %8 = subi %1, %c1 : index
    %9 = addi %8, %c1 : index
    %10 = alloca() : memref<1xf64>
    call @S15(%10, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    %11 = alloca() : memref<1xf64>
    call @S16(%11, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to %1 {
      call @S17(%0, %arg4, %arg5) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg6 = 1 to %1 {
        call @S18(%0, %arg3, %arg6, %arg1, %arg5, %10) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
      }
      call @S19(%arg3, %arg5, %arg1, %11) : (memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
    }
    return
  }
  func @S0(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S2(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S3(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S4(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S5(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg1[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %c0 = constant 0 : index
    affine.store %4, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S6(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf64>
    %1 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
  func @S7(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S8(%arg0: memref<1xf64>, %arg1: memref<2000x2000xf64>, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg1[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %c0 = constant 0 : index
    affine.store %4, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S9(%arg0: memref<2000x2000xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf64>
    affine.store %0, %arg0[%arg1, %arg2] : memref<2000x2000xf64>
    return
  }
  func @S10(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S11(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S12(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[%arg2] : memref<2000xf64>
    %c0 = constant 0 : index
    affine.store %0, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S13(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.load %arg3[%arg4, %arg2] : memref<2000x2000xf64>
    %2 = affine.load %arg1[%arg2] : memref<2000xf64>
    %3 = mulf %1, %2 : f64
    %4 = subf %0, %3 : f64
    %c0 = constant 0 : index
    affine.store %4, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S14(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    affine.store %0, %arg0[%arg1] : memref<2000xf64>
    return
  }
  func @S15(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S16(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = affine.load %arg1[%c0] : memref<1xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S17(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.apply #map1(%arg2)
    %1 = affine.load %arg1[%0] : memref<2000xf64>
    %c0 = constant 0 : index
    affine.store %1, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S18(%arg0: memref<1xf64>, %arg1: memref<2000xf64>, %arg2: index, %arg3: memref<2000x2000xf64>, %arg4: index, %arg5: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg5[0] : memref<1xf64>
    %1 = affine.apply #map1(%arg4)
    %2 = affine.load %arg3[%1, %arg2] : memref<2000x2000xf64>
    %3 = affine.load %arg1[%arg2] : memref<2000xf64>
    %4 = mulf %2, %3 : f64
    %5 = subf %0, %4 : f64
    %c0 = constant 0 : index
    affine.store %5, %arg0[%c0] : memref<1xf64>
    return
  }
  func @S19(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000x2000xf64>, %arg3: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg3[0] : memref<1xf64>
    %1 = affine.apply #map1(%arg1)
    %2 = affine.load %arg2[%1, %1] : memref<2000x2000xf64>
    %3 = divf %0, %2 : f64
    affine.store %3, %arg0[%1] : memref<2000xf64>
    return
  }
  func @kernel_ludcmp_new(%arg0: memref<2000xf64>, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: i32, %arg4: memref<2000x2000xf64>) {
    %0 = alloca() : memref<1xf64>
    %1 = alloca() : memref<1xf64>
    %2 = alloca() : memref<1xf64>
    %3 = alloca() : memref<1xf64>
    %4 = alloca() : memref<1xf64>
    %5 = alloca() : memref<1xf64>
    %6 = alloca() : memref<1xf64>
    %7 = alloca() : memref<1xf64>
    %8 = alloca() : memref<1xf64>
    %9 = index_cast %arg3 : i32 to index
    %10 = alloca() : memref<1xf64>
    %11 = alloca() : memref<1xf64>
    %12 = alloca() : memref<1xf64>
    call @S3(%4, %8) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S2(%5, %8) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S1(%6, %8) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S0(%7, %8) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to %9 {
      %13 = alloca() : memref<1xf64>
      %14 = alloca() : memref<1xf64>
      %c0_0 = constant 0 : index
      %15 = affine.apply #map1(%arg5)
      call @S7(%14, %arg4, %c0_0, %15) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S9(%arg4, %c0_0, %15, %13) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    }
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    call @S4(%2, %arg4, %c1, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
    call @S6(%arg4, %c1, %c0, %6) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
    affine.for %arg5 = 1 to %9 {
      %13 = alloca() : memref<1xf64>
      %14 = alloca() : memref<1xf64>
      %15 = alloca() : memref<1xf64>
      %16 = affine.apply #map1(%arg5)
      call @S7(%15, %arg4, %c1, %16) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S9(%arg4, %c1, %16, %14) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      call @S8(%15, %arg4, %c1, %16, %c0, %13) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index, memref<1xf64>) -> ()
    }
    affine.for %arg5 = 2 to %9 {
      %13 = alloca() : memref<1xf64>
      %14 = alloca() : memref<1xf64>
      %15 = affine.apply #map1(%arg5)
      call @S4(%14, %arg4, %15, %c0) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
      call @S6(%arg4, %15, %c0, %13) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
      affine.for %arg6 = 1 to #map1(%arg5) {
        %16 = alloca() : memref<1xf64>
        %17 = alloca() : memref<1xf64>
        %18 = affine.apply #map1(%arg6)
        call @S4(%17, %arg4, %15, %18) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        call @S6(%arg4, %15, %18, %16) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
        affine.for %arg7 = 0 to #map1(%arg6) {
          %19 = alloca() : memref<1xf64>
          %20 = alloca() : memref<1xf64>
          %21 = affine.apply #map1(%arg7)
          call @S5(%20, %arg4, %15, %18, %21, %19) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index, memref<1xf64>) -> ()
        }
      }
      affine.for %arg6 = #map1(%arg5) to %9 {
        %16 = alloca() : memref<1xf64>
        %17 = alloca() : memref<1xf64>
        %18 = affine.apply #map1(%arg6)
        call @S7(%17, %arg4, %15, %18) : (memref<1xf64>, memref<2000x2000xf64>, index, index) -> ()
        call @S9(%arg4, %15, %18, %16) : (memref<2000x2000xf64>, index, index, memref<1xf64>) -> ()
        affine.for %arg7 = 0 to #map1(%arg5) {
          %19 = alloca() : memref<1xf64>
          %20 = alloca() : memref<1xf64>
          %21 = affine.apply #map1(%arg7)
          call @S8(%20, %arg4, %15, %18, %21, %19) : (memref<1xf64>, memref<2000x2000xf64>, index, index, index, memref<1xf64>) -> ()
        }
      }
    }
    call @S11(%10, %1) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to #map9()[%9] {
      affine.for %arg6 = #map7(%arg5) to min #map8(%arg5)[%9] {
        %13 = alloca() : memref<1xf64>
        %14 = affine.apply #map1(%arg6)
        call @S14(%arg0, %14, %13) : (memref<2000xf64>, index, memref<1xf64>) -> ()
      }
    }
    call @S10(%3, %1) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S12(%1, %arg2, %c0) : (memref<1xf64>, memref<2000xf64>, index) -> ()
    affine.for %arg5 = 1 to %9 {
      %13 = alloca() : memref<1xf64>
      %14 = affine.apply #map1(%arg5)
      call @S12(%13, %arg2, %14) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      affine.for %arg6 = 0 to #map1(%arg5) {
        %15 = alloca() : memref<1xf64>
        %16 = alloca() : memref<1xf64>
        %17 = affine.apply #map1(%arg6)
        call @S13(%16, %arg0, %14, %arg4, %17, %15) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
      }
    }
    call @S16(%12, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    call @S15(%11, %0) : (memref<1xf64>, memref<1xf64>) -> ()
    affine.for %arg5 = 0 to %9 {
      affine.if #set0(%arg5) {
        %13 = alloca() : memref<1xf64>
        %14 = alloca() : memref<1xf64>
        call @S19(%arg1, %c0, %arg4, %14) : (memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
        call @S17(%13, %arg0, %c0) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      }
      affine.if #set1(%arg5) {
        %13 = alloca() : memref<1xf64>
        %14 = affine.apply #map1(%arg5)
        call @S17(%13, %arg0, %14) : (memref<1xf64>, memref<2000xf64>, index) -> ()
      }
      affine.for %arg6 = 1 to #map1(%arg5) {
        %13 = alloca() : memref<1xf64>
        %14 = alloca() : memref<1xf64>
        %15 = affine.apply #map1(%arg5)
        %16 = affine.apply #map1(%arg6)
        call @S18(%14, %arg1, %15, %arg4, %16, %13) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
      }
      affine.if #set1(%arg5) {
        %13 = alloca() : memref<1xf64>
        %14 = alloca() : memref<1xf64>
        %15 = alloca() : memref<1xf64>
        %16 = affine.apply #map1(%arg5)
        call @S18(%15, %arg1, %16, %arg4, %16, %14) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
        call @S19(%arg1, %16, %arg4, %13) : (memref<2000xf64>, index, memref<2000x2000xf64>, memref<1xf64>) -> ()
      }
      affine.for %arg6 = #map10(%arg5) to %9 {
        %13 = alloca() : memref<1xf64>
        %14 = alloca() : memref<1xf64>
        %15 = affine.apply #map1(%arg5)
        %16 = affine.apply #map1(%arg6)
        call @S18(%14, %arg1, %15, %arg4, %16, %13) : (memref<1xf64>, memref<2000xf64>, index, memref<2000x2000xf64>, index, memref<1xf64>) -> ()
      }
    }
    return
  }
}
