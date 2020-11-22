#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1 - 1)>
#map7 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map8 = affine_map<(d0, d1) -> (d0 + 1, d1 - 1)>

#set0 = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
#set2 = affine_set<(d0, d1) : (d1 - d0 - 2 >= 0)>

module {
  func @max_score(%arg0: i32, %arg1: i32) -> i32 {
    %0 = cmpi "sge", %arg0, %arg1 : i32
    %1 = select %0, %arg0, %arg1 : i32
    return %1 : i32
  }
  func @match(%arg0: i8, %arg1: i8) -> i32 {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c3_i8 = constant 3 : i8
    %0 = addi %arg0, %arg1 : i8
    %1 = cmpi "eq", %0, %c3_i8 : i8
    %2 = select %1, %c1_i32, %c0_i32 : i32
    return %2 : i32
  }
  func @pb_nussinov(%arg0: memref<?xi8>, %arg1: memref<?x?xi32>) {
    %c0 = constant 0 : index
    %0 = dim %arg0, %c0 : memref<?xi8>
    affine.for %arg2 = 0 to %0 {
      %1 = affine.apply #map0(%arg2)[%0]
      affine.for %arg3 = #map1(%1) to %0 {
        affine.if #set0(%arg3) {
          call @S0(%arg1, %arg3, %arg2, %arg0) : (memref<?x?xi32>, index, index, memref<?xi8>) -> ()
        }
        affine.if #set1(%1)[%0] {
          call @S1(%arg1, %arg3, %arg2, %arg0) : (memref<?x?xi32>, index, index, memref<?xi8>) -> ()
        }
        affine.if #set0(%arg3) {
          affine.if #set1(%1)[%0] {
            affine.if #set2(%1, %arg3) {
              call @S2(%arg1, %arg3, %arg2, %arg0) : (memref<?x?xi32>, index, index, memref<?xi8>) -> ()
            } else {
              call @S3(%arg1, %arg3, %arg2, %arg0) : (memref<?x?xi32>, index, index, memref<?xi8>) -> ()
            }
          }
        }
        affine.for %arg4 = #map1(%1) to #map2(%arg3) {
          call @S4(%arg1, %arg3, %arg2, %arg0, %arg4) : (memref<?x?xi32>, index, index, memref<?xi8>, index) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: memref<?xi8>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = dim %arg3, %c0 : memref<?xi8>
    %1 = affine.apply #map0(%arg2)[%0]
    %2 = affine.load %arg0[%1, %arg1] : memref<?x?xi32>
    %3 = affine.load %arg0[%1, %arg1 - 1] : memref<?x?xi32>
    %4 = call @max_score(%2, %3) : (i32, i32) -> i32
    affine.store %4, %arg0[%1, %arg1] : memref<?x?xi32>
    return
  }
  func @S1(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: memref<?xi8>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = dim %arg3, %c0 : memref<?xi8>
    %1 = affine.apply #map0(%arg2)[%0]
    %2 = affine.load %arg0[%1, %arg1] : memref<?x?xi32>
    %3 = affine.load %arg0[%1 + 1, %arg1] : memref<?x?xi32>
    %4 = call @max_score(%2, %3) : (i32, i32) -> i32
    affine.store %4, %arg0[%1, %arg1] : memref<?x?xi32>
    return
  }
  func @S2(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: memref<?xi8>) attributes {scop.stmt} {
    %0 = affine.load %arg3[%arg1] : memref<?xi8>
    %c0 = constant 0 : index
    %1 = dim %arg3, %c0 : memref<?xi8>
    %2 = affine.apply #map0(%arg2)[%1]
    %3 = affine.load %arg3[%2] : memref<?xi8>
    %4 = call @match(%3, %0) : (i8, i8) -> i32
    %5 = affine.load %arg0[%2 + 1, %arg1 - 1] : memref<?x?xi32>
    %6 = addi %5, %4 : i32
    %7 = affine.load %arg0[%2, %arg1] : memref<?x?xi32>
    %8 = call @max_score(%7, %6) : (i32, i32) -> i32
    affine.store %8, %arg0[%2, %arg1] : memref<?x?xi32>
    return
  }
  func @S3(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: memref<?xi8>) attributes {scop.stmt} {
    %c0 = constant 0 : index
    %0 = dim %arg3, %c0 : memref<?xi8>
    %1 = affine.apply #map0(%arg2)[%0]
    %2 = affine.load %arg0[%1, %arg1] : memref<?x?xi32>
    %3 = affine.load %arg0[%1 + 1, %arg1 - 1] : memref<?x?xi32>
    %4 = call @max_score(%2, %3) : (i32, i32) -> i32
    affine.store %4, %arg0[%1, %arg1] : memref<?x?xi32>
    return
  }
  func @S4(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: memref<?xi8>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg4 + 1, %arg1] : memref<?x?xi32>
    %c0 = constant 0 : index
    %1 = dim %arg3, %c0 : memref<?xi8>
    %2 = affine.apply #map0(%arg2)[%1]
    %3 = affine.load %arg0[%2, %arg4] : memref<?x?xi32>
    %4 = addi %3, %0 : i32
    %5 = affine.load %arg0[%2, %arg1] : memref<?x?xi32>
    %6 = call @max_score(%5, %4) : (i32, i32) -> i32
    affine.store %6, %arg0[%2, %arg1] : memref<?x?xi32>
    return
  }
}
