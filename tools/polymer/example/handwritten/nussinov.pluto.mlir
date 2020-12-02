#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<()[s0] -> ((s0 - 62) floordiv 32 + 1)>
#map4 = affine_map<(d0)[s0] -> (d0 * -32 + s0 - 31)>
#map5 = affine_map<(d0) -> (d0 * 32 + 31)>
#map6 = affine_map<()[s0] -> (0, (s0 - 61) ceildiv 32)>
#map7 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>
#map8 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map9 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map10 = affine_map<()[s0] -> (s0 - 1)>
#map11 = affine_map<(d0, d1)[s0] -> (2, d0 * 32 - d1 * 32, d1 * -32 + s0 - 30)>
#map12 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map13 = affine_map<(d0, d1)[s0] -> (d0 * 32, -d1 + s0 + 1)>
#map14 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#set0 = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
#set2 = affine_set<(d0, d1) : (d1 - d0 - 2 >= 0)>
#set3 = affine_set<()[s0] : (s0 - 62 >= 0)>
#set4 = affine_set<()[s0] : ((s0 + 2) mod 32 == 0)>
#set5 = affine_set<(d0, d1)[s0] : (-d0 + (s0 - 31) floordiv 32 >= 0, -d1 + s0 floordiv 32 - 1 >= 0)>
#set6 = affine_set<(d0, d1)[s0] : (d0 - d1 == 0, d0 - (s0 - 31) ceildiv 32 >= 0)>
#set7 = affine_set<(d0, d1)[s0] : ((-d1 + s0) floordiv 32 - d0 >= 0)>
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
      affine.for %arg3 = #map1(%arg2)[%0] to %0 {
        affine.if #set0(%arg3) {
          call @S0(%arg1, %arg3, %arg2, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
        affine.if #set1(%1)[%0] {
          call @S1(%arg1, %arg3, %arg2, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
        affine.if #set0(%arg3) {
          affine.if #set1(%1)[%0] {
            affine.if #set2(%1, %arg3) {
              call @S2(%arg1, %arg3, %arg2, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
            } else {
              call @S3(%arg1, %arg3, %arg2, %0) : (memref<?x?xi32>, index, index, index) -> ()
            }
          }
        }
        affine.for %arg4 = #map1(%arg2)[%0] to #map2(%arg3) {
          call @S4(%arg1, %arg3, %arg2, %0, %arg4) : (memref<?x?xi32>, index, index, index, index) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    %1 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1) - 1] : memref<?x?xi32>
    %2 = call @max_score(%0, %1) : (i32, i32) -> i32
    affine.store %2, %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    return
  }
  func @S1(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    %1 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3), symbol(%arg1)] : memref<?x?xi32>
    %2 = call @max_score(%0, %1) : (i32, i32) -> i32
    affine.store %2, %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    return
  }
  func @S2(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?xi8>) attributes {scop.stmt} {
    %0 = affine.load %arg4[symbol(%arg1)] : memref<?xi8>
    %1 = affine.load %arg4[-symbol(%arg2) + symbol(%arg3) - 1] : memref<?xi8>
    %2 = call @match(%1, %0) : (i8, i8) -> i32
    %3 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3), symbol(%arg1) - 1] : memref<?x?xi32>
    %4 = addi %3, %2 : i32
    %5 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    %6 = call @max_score(%5, %4) : (i32, i32) -> i32
    affine.store %6, %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    return
  }
  func @S3(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    %1 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3), symbol(%arg1) - 1] : memref<?x?xi32>
    %2 = call @max_score(%0, %1) : (i32, i32) -> i32
    affine.store %2, %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    return
  }
  func @S4(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg4) + 1, symbol(%arg1)] : memref<?x?xi32>
    %1 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg4)] : memref<?x?xi32>
    %2 = addi %1, %0 : i32
    %3 = affine.load %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    %4 = call @max_score(%3, %2) : (i32, i32) -> i32
    affine.store %4, %arg0[-symbol(%arg2) + symbol(%arg3) - 1, symbol(%arg1)] : memref<?x?xi32>
    return
  }
  func @pb_nussinov_new(%arg0: memref<?xi8>, %arg1: memref<?x?xi32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg0, %c0 : memref<?xi8>
    affine.if #set3()[%0] {
      affine.if #set4()[%0] {
        affine.for %arg2 = 0 to #map3()[%0] {
          %1 = affine.apply #map4(%arg2)[%0]
          %2 = affine.apply #map5(%arg2)
          call @S0(%arg1, %1, %2, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %3 = affine.apply #map4(%arg2)[%0]
          %4 = affine.apply #map5(%arg2)
          call @S1(%arg1, %3, %4, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %5 = affine.apply #map4(%arg2)[%0]
          %6 = affine.apply #map5(%arg2)
          call @S2(%arg1, %5, %6, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
          %7 = affine.apply #map4(%arg2)[%0]
          %8 = affine.apply #map5(%arg2)
          call @S3(%arg1, %7, %8, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
      }
    }
    affine.for %arg2 = max #map6()[%0] to #map7()[%0] {
      affine.for %arg3 = max #map8(%arg2)[%0] to min #map9(%arg2)[%0] {
        affine.if #set5(%arg2, %arg3)[%0] {
          %1 = affine.apply #map4(%arg3)[%0]
          %2 = affine.apply #map5(%arg3)
          call @S0(%arg1, %1, %2, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %3 = affine.apply #map4(%arg3)[%0]
          %4 = affine.apply #map5(%arg3)
          call @S1(%arg1, %3, %4, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %5 = affine.apply #map4(%arg3)[%0]
          %6 = affine.apply #map5(%arg3)
          call @S2(%arg1, %5, %6, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
          %7 = affine.apply #map4(%arg3)[%0]
          %8 = affine.apply #map5(%arg3)
          call @S3(%arg1, %7, %8, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
        affine.if #set6(%arg2, %arg3)[%0] {
          %1 = affine.apply #map10()[%0]
          call @S0(%arg1, %c1, %1, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %2 = affine.apply #map10()[%0]
          call @S1(%arg1, %c1, %2, %0) : (memref<?x?xi32>, index, index, index) -> ()
          %3 = affine.apply #map10()[%0]
          call @S2(%arg1, %c1, %3, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
          %4 = affine.apply #map10()[%0]
          call @S3(%arg1, %c1, %4, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
        affine.for %arg4 = max #map11(%arg2, %arg3)[%0] to min #map12(%arg2, %arg3)[%0] {
          affine.if #set7(%arg3, %arg4)[%0] {
            %1 = affine.apply #map1(%arg4)[%0]
            call @S0(%arg1, %arg4, %1, %0) : (memref<?x?xi32>, index, index, index) -> ()
            %2 = affine.apply #map1(%arg4)[%0]
            call @S1(%arg1, %arg4, %2, %0) : (memref<?x?xi32>, index, index, index) -> ()
            %3 = affine.apply #map1(%arg4)[%0]
            call @S2(%arg1, %arg4, %3, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
            %4 = affine.apply #map1(%arg4)[%0]
            call @S3(%arg1, %arg4, %4, %0) : (memref<?x?xi32>, index, index, index) -> ()
          }
          affine.for %arg5 = max #map13(%arg3, %arg4)[%0] to min #map14(%arg3)[%0] {
            call @S0(%arg1, %arg4, %arg5, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S1(%arg1, %arg4, %arg5, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S2(%arg1, %arg4, %arg5, %0, %arg0) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
            call @S3(%arg1, %arg4, %arg5, %0) : (memref<?x?xi32>, index, index, index) -> ()
            affine.for %arg6 = #map1(%arg4)[%0] to #map2(%arg5) {
              call @S4(%arg1, %arg4, %arg5, %0, %arg6) : (memref<?x?xi32>, index, index, index, index) -> ()
            }
          }
        }
      }
    }
    return
  }
}

