#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
#set0 = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
#set2 = affine_set<(d0, d1) : (d1 - d0 - 2 >= 0)>
module  {
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
          %2 = affine.load %arg1[%1, %arg3] : memref<?x?xi32>
          %3 = affine.load %arg1[%1, %arg3 - 1] : memref<?x?xi32>
          %4 = call @max_score(%2, %3) : (i32, i32) -> i32
          affine.store %4, %arg1[%1, %arg3] : memref<?x?xi32>
        }
        affine.if #set1(%1)[%0] {
          %2 = affine.load %arg1[%1, %arg3] : memref<?x?xi32>
          %3 = affine.load %arg1[%1 + 1, %arg3] : memref<?x?xi32>
          %4 = call @max_score(%2, %3) : (i32, i32) -> i32
          affine.store %4, %arg1[%1, %arg3] : memref<?x?xi32>
        }
        affine.if #set0(%arg3) {
          affine.if #set1(%1)[%0] {
            affine.if #set2(%1, %arg3) {
              %2 = affine.load %arg0[%1] : memref<?xi8>
              %3 = affine.load %arg0[%arg3] : memref<?xi8>
              %4 = call @match(%2, %3) : (i8, i8) -> i32
              %5 = affine.load %arg1[%1 + 1, %arg3 - 1] : memref<?x?xi32>
              %6 = addi %5, %4 : i32
              %7 = affine.load %arg1[%1, %arg3] : memref<?x?xi32>
              %8 = call @max_score(%7, %6) : (i32, i32) -> i32
              affine.store %8, %arg1[%1, %arg3] : memref<?x?xi32>
            } else {
              %2 = affine.load %arg1[%1, %arg3] : memref<?x?xi32>
              %3 = affine.load %arg1[%1 + 1, %arg3 - 1] : memref<?x?xi32>
              %4 = call @max_score(%2, %3) : (i32, i32) -> i32
              affine.store %4, %arg1[%1, %arg3] : memref<?x?xi32>
            }
          }
        }
        affine.for %arg4 = #map1(%1) to #map2(%arg3) {
          %2 = affine.load %arg1[%1, %arg4] : memref<?x?xi32>
          %3 = affine.load %arg1[%arg4 + 1, %arg3] : memref<?x?xi32>
          %4 = addi %2, %3 : i32
          %5 = affine.load %arg1[%1, %arg3] : memref<?x?xi32>
          %6 = call @max_score(%5, %4) : (i32, i32) -> i32
          affine.store %6, %arg1[%1, %arg3] : memref<?x?xi32>
        }
      }
    }
    return
  }
}

