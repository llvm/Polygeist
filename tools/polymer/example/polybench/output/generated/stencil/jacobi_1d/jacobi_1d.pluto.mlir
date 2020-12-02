#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<()[s0] -> (s0 + 1)>
#map5 = affine_map<(d0) -> (d0 * 32 + 1)>
#map6 = affine_map<(d0) -> (d0 * 32 + 3)>
#map7 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 * 32 + s0 * 4 - 3)>
#map8 = affine_map<(d0, d1)[s0, s1] -> (d0 * 32 + 32, s0 * 2 + s1 - 3, d1 * -32 + d0 * 32 + s0 * 4 - 1)>
#map9 = affine_map<(d0)[s0] -> (d0 ceildiv 2, (d0 * 16 - s0 + 2) ceildiv 16)>
#map10 = affine_map<(d0)[s0, s1] -> ((s0 * 2 + s1 - 4) floordiv 32 + 1, (d0 * 16 + s1 + 13) floordiv 32 + 1, (d0 * 32 + s1 + 28) floordiv 32 + 1)>
#map11 = affine_map<() -> (-1)>
#map12 = affine_map<()[s0, s1] -> ((s0 - 1) floordiv 8 + 1, (s0 * 4 + s1 - 8) floordiv 32 + 1)>

#set0 = affine_set<()[s0] : ((s0 + 15) mod 16 == 0)>
#set1 = affine_set<(d0)[s0] : (d0 * 8 - (s0 - 1) == 0)>
#set2 = affine_set<(d0) : (d0 + 1 == 0)>
#set3 = affine_set<(d0, d1)[s0, s1] : ((d1 * 32 + s0 * 2 - s1 + 1) floordiv 32 - d0 >= 0, d0 - (d1 * 32 + s0 * 2 - s1 - 30) ceildiv 32 >= 0, d1 - (s0 * 2 + s1 - 34) ceildiv 32 >= 0)>
#set4 = affine_set<()[s0] : ((s0 + 29) mod 32 == 0)>
#set5 = affine_set<()[s0, s1] : ((s0 * 2 + s1 + 29) mod 32 == 0)>
#set6 = affine_set<(d0)[s0] : (d0 - (s0 - 8) ceildiv 8 >= 0)>
#set7 = affine_set<()[s0] : ((s0 + 7) mod 8 == 0)>
#set8 = affine_set<()[s0] : (s0 - 3 == 0)>

module {
  func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %0 {
      affine.for %arg5 = 1 to #map1()[%1] {
        call @S0(%arg3, %arg5, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
      affine.for %arg5 = 1 to #map1()[%1] {
        call @S1(%arg2, %arg5, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
      }
    }
    return
  }
  func @S0(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[symbol(%arg1) - 1] : memref<2000xf64>
    %1 = affine.load %arg2[symbol(%arg1)] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[symbol(%arg1) + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @S1(%arg0: memref<2000xf64>, %arg1: index, %arg2: memref<2000xf64>) attributes {scop.stmt} {
    %cst = constant 3.333300e-01 : f64
    %0 = affine.load %arg2[symbol(%arg1) - 1] : memref<2000xf64>
    %1 = affine.load %arg2[symbol(%arg1)] : memref<2000xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg2[symbol(%arg1) + 1] : memref<2000xf64>
    %4 = addf %2, %3 : f64
    %5 = mulf %cst, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<2000xf64>
    return
  }
  func @kernel_jacobi_1d_new(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
    %c0 = constant 0 : index
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg4 = -1 to min #map12()[%1, %0] {
      affine.if #set1(%arg4)[%1] {
        affine.if #set0()[%1] {
          %2 = affine.apply #map1()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.for %arg5 = max #map9(%arg4)[%1] to min #map10(%arg4)[%1, %0] {
        affine.if #set2(%arg4) {
          affine.for %arg6 = #map5(%arg5) to #map6(%arg5) {
            call @S0(%arg3, %c0, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
          }
        }
        affine.for %arg6 = #map7(%arg4, %arg5)[%1] to min #map8(%arg4, %arg5)[%1, %0] {
          %2 = affine.apply #map1()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
        affine.if #set3(%arg4, %arg5)[%1, %0] {
          %2 = affine.apply #map1()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set2(%arg4) {
        affine.if #set4()[%0] {
          call @S0(%arg3, %c0, %arg2) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
      affine.if #set6(%arg4)[%1] {
        affine.if #set5()[%1, %0] {
          %2 = affine.apply #map1()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    affine.if #set8()[%0] {
      affine.if #set7()[%1] {
        affine.if #set0()[%1] {
          %2 = affine.apply #map1()[%1]
          call @S1(%arg2, %2, %arg3) : (memref<2000xf64>, index, memref<2000xf64>) -> ()
        }
      }
    }
    return
  }
}
