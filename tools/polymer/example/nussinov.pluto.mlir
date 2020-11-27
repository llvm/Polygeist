#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1 - 1)>
#map7 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map8 = affine_map<(d0, d1) -> (d0 + 1, d1 - 1)>
#map9 = affine_map<()[s0] -> ((s0 - 62) floordiv 32 + 1)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 * 32, -d1 + s0)>
#map11 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map12 = affine_map<(d0, d1, d2)[s0] -> (1, d0 * 32 - d1 * 32, d2 * -32 + s0 - 31)>
#map13 = affine_map<()[s0] -> (s0 - 31)>
#map14 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map15 = affine_map<(d0) -> (32, d0)>
#map16 = affine_map<(d0, d1)[s0] -> (d0 * 32, -d1 + s0 + 1)>
#map17 = affine_map<(d0, d1, d2)[s0, s1] -> (2, s0 - 31, d0 * 32 - d1 * 32, d2 * -32 + s1 - 30)>
#map18 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#map19 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32 + 32)>
#map20 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1)>
#map21 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 + 1, -d2 + s0 + 1)>
#map22 = affine_map<(d0, d1, d2, d3)[s0, s1] -> (2, d0 * 32 - d1 * 32, d2 * -32 + s0 - 30, d3 * -32 + s1 - 31)>
#map23 = affine_map<(d0, d1)[s0] -> (1, (d0 * -32 + d1 * 32 + s0 - 62) ceildiv 32)>
#map24 = affine_map<(d0)[s0] -> ((s0 - 2) floordiv 32 + 1, d0 + 1)>
#map25 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map26 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map27 = affine_map<()[s0] -> (0, (s0 - 61) ceildiv 32)>
#map28 = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>

#set0 = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
#set2 = affine_set<(d0, d1) : (d1 - d0 - 2 >= 0)>
#set3 = affine_set<()[s0] : ((s0 + 2) mod 32 == 0)>
#set4 = affine_set<()[s0] : (s0 - 62 == 0)>
#set5 = affine_set<(d0, d1)[s0, s1] : (s0 - 32 == 0, d0 - (s1 - 31) floordiv 32 == 0, d1 == 0)>
#set6 = affine_set<(d0, d1)[s0] : (s0 - 31 == 0, d0 == 0, d1 == 0)>
#set7 = affine_set<(d0) : (d0 == 0)>
#set8 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 32 + s0 - 62) ceildiv 32 == 0)>
#set9 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 32 + s0 - 63) floordiv 32 == 0)>

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
        affine.for %arg4 = #map1(%1) to #map2(%arg3) {
          call @S4(%arg1, %arg3, %arg2, %0, %arg4) : (memref<?x?xi32>, index, index, index, index) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map0(%arg2)[%arg3]
    %1 = affine.load %arg0[%0, %arg1] : memref<?x?xi32>
    %2 = affine.load %arg0[%0, %arg1 - 1] : memref<?x?xi32>
    %3 = call @max_score(%1, %2) : (i32, i32) -> i32
    affine.store %3, %arg0[%0, %arg1] : memref<?x?xi32>
    return
  }
  func @S1(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map0(%arg2)[%arg3]
    %1 = affine.load %arg0[%0, %arg1] : memref<?x?xi32>
    %2 = affine.load %arg0[%0 + 1, %arg1] : memref<?x?xi32>
    %3 = call @max_score(%1, %2) : (i32, i32) -> i32
    affine.store %3, %arg0[%0, %arg1] : memref<?x?xi32>
    return
  }
  func @S2(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?xi8>) attributes {scop.stmt} {
    %0 = affine.load %arg4[%arg1] : memref<?xi8>
    %1 = affine.apply #map0(%arg2)[%arg3]
    %2 = affine.load %arg4[%1] : memref<?xi8>
    %3 = call @match(%2, %0) : (i8, i8) -> i32
    %4 = affine.load %arg0[%1 + 1, %arg1 - 1] : memref<?x?xi32>
    %5 = addi %4, %3 : i32
    %6 = affine.load %arg0[%1, %arg1] : memref<?x?xi32>
    %7 = call @max_score(%6, %5) : (i32, i32) -> i32
    affine.store %7, %arg0[%1, %arg1] : memref<?x?xi32>
    return
  }
  func @S3(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) attributes {scop.stmt} {
    %0 = affine.apply #map0(%arg2)[%arg3]
    %1 = affine.load %arg0[%0, %arg1] : memref<?x?xi32>
    %2 = affine.load %arg0[%0 + 1, %arg1 - 1] : memref<?x?xi32>
    %3 = call @max_score(%1, %2) : (i32, i32) -> i32
    affine.store %3, %arg0[%0, %arg1] : memref<?x?xi32>
    return
  }
  func @S4(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[%arg4 + 1, %arg1] : memref<?x?xi32>
    %1 = affine.apply #map0(%arg2)[%arg3]
    %2 = affine.load %arg0[%1, %arg4] : memref<?x?xi32>
    %3 = addi %2, %0 : i32
    %4 = affine.load %arg0[%1, %arg1] : memref<?x?xi32>
    %5 = call @max_score(%4, %3) : (i32, i32) -> i32
    affine.store %5, %arg0[%1, %arg1] : memref<?x?xi32>
    return
  }
  func @pb_nussinov_new(%arg0: memref<?x?xi32>, %arg1: memref<?xi8>) {
    %c0 = constant 0 : index
    %0 = dim %arg1, %c0 : memref<?xi8>
    affine.if #set4()[%0] {
      affine.if #set3()[%0] {
        affine.for %arg2 = 0 to #map9()[%0] {
          call @S0(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
          call @S1(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
          call @S2(%arg0, %c0, %c0, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
          call @S3(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
        }
      }
    }
    affine.for %arg2 = max #map27()[%0] to #map28()[%0] {
      affine.for %arg3 = max #map25(%arg2)[%0] to min #map26(%arg2)[%0] {
        affine.if #set8(%arg2, %arg3)[%0] {
          affine.for %arg4 = max #map12(%arg2, %arg3, %arg3)[%0] to #map13()[%0] {
            affine.for %arg5 = max #map10(%arg3, %arg4)[%0] to min #map11(%arg3)[%0] {
              call @S0(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S1(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S2(%arg0, %arg5, %arg4, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
              call @S3(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
            }
          }
          affine.if #set5(%arg2, %arg3)[%0, %0] {
            call @S0(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S1(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S2(%arg0, %c0, %c0, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
            call @S3(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
          }
          affine.if #set6(%arg2, %arg3)[%0] {
            call @S0(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S1(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
            call @S2(%arg0, %c0, %c0, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
            call @S3(%arg0, %c0, %c0, %0) : (memref<?x?xi32>, index, index, index) -> ()
          }
          affine.for %arg4 = max #map17(%arg2, %arg3, %arg3)[%0, %0] to min #map18(%arg2, %arg3)[%0] {
            affine.if #set7(%arg3) {
              call @S0(%arg0, %c0, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S1(%arg0, %c0, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S2(%arg0, %c0, %arg4, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
              call @S3(%arg0, %c0, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
            }
            affine.for %arg5 = max #map16(%arg3, %arg4)[%0] to min #map11(%arg3)[%0] {
              call @S0(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S1(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S2(%arg0, %arg5, %arg4, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
              call @S3(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              affine.for %arg6 = #map14(%arg4)[%0] to min #map15(%arg5) {
                call @S4(%arg0, %arg5, %arg4, %0, %arg6) : (memref<?x?xi32>, index, index, index, index) -> ()
              }
            }
          }
        }
        affine.if #set9(%arg2, %arg3)[%0] {
          affine.for %arg4 = max #map12(%arg2, %arg3, %arg3)[%0] to #map19(%arg2, %arg3) {
            affine.for %arg5 = max #map10(%arg3, %arg4)[%0] to min #map11(%arg3)[%0] {
              call @S0(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S1(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
              call @S2(%arg0, %arg5, %arg4, %0, %arg1) : (memref<?x?xi32>, index, index, index, memref<?xi8>) -> ()
              call @S3(%arg0, %arg5, %arg4, %0) : (memref<?x?xi32>, index, index, index) -> ()
            }
          }
        }
        affine.for %arg4 = max #map23(%arg2, %arg3)[%0] to min #map24(%arg3)[%0] {
          affine.for %arg5 = max #map22(%arg2, %arg3, %arg3, %arg4)[%0, %0] to min #map18(%arg2, %arg3)[%0] {
            affine.for %arg6 = max #map21(%arg3, %arg4, %arg5)[%0] to min #map11(%arg3)[%0] {
              affine.for %arg7 = max #map10(%arg4, %arg5)[%0] to min #map20(%arg4, %arg6) {
                call @S4(%arg0, %arg6, %arg5, %0, %arg7) : (memref<?x?xi32>, index, index, index, index) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
// [pluto] compute_deps (isl)
// [pluto] Diamond tiling not possible/useful
// [Pluto] After tiling:
// T(S1): (i0/32, i1/32, 0, i0, i1, 0, 1)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S2): (i0/32, i1/32, 0, i0, i1, 0, 2)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S3): (i0/32, i1/32, 0, i0, i1, 0, 3)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S4): (i0/32, i1/32, 0, i0, i1, 0, 4)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S5): (i0/32, i1/32, i2/32, i0, i1, i2, 0)
// loop types (loop, loop, loop, loop, loop, loop, scalar)

// [Pluto] After tile scheduling:
// T(S1): (i0/32+i1/32, i1/32, 0, i0, i1, 0, 1)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S2): (i0/32+i1/32, i1/32, 0, i0, i1, 0, 2)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S3): (i0/32+i1/32, i1/32, 0, i0, i1, 0, 3)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S4): (i0/32+i1/32, i1/32, 0, i0, i1, 0, 4)
// loop types (loop, loop, scalar, loop, loop, scalar, scalar)

// T(S5): (i0/32+i1/32, i1/32, i2/32, i0, i1, i2, 0)
// loop types (loop, loop, loop, loop, loop, loop, scalar)

// if (P0 >= 62) {
//   if ((P0+2)%32 == 0) {
//     for (t2=0;t2<=floord(P0-62,32);t2++) {
//       S0()
//       S1()
//       S2()
//       S3()
//     }
//   }
// }
// for (t1=max(0,ceild(P0-61,32));t1<=floord(P0-1,16);t1++) {
//   for (t2=max(0,ceild(32*t1-P0+1,32));t2<=min(floord(P0-1,32),t1);t2++) {
//     if (t1 >= ceild(32*t2+P0-62,32)) {
//       for (t4=max(max(1,32*t1-32*t2),-32*t2+P0-31);t4<=P0-32;t4++) {
//         for (t5=max(32*t2,-t4+P0);t5<=min(P0-1,32*t2+31);t5++) {
//           S0()
//           S1()
//           S2()
//           S3()
//         }
//       }
//       if ((P0 >= 32) && (t1 <= floord(P0-31,32)) && (t2 == 0)) {
//         S0()
//         S1()
//         S2()
//         S3()
//       }
//       if ((P0 <= 31) && (t1 == 0) && (t2 == 0)) {
//         S0()
//         S1()
//         S2()
//         S3()
//       }
//       for (t4=max(max(max(2,P0-31),32*t1-32*t2),-32*t2+P0-30);t4<=min(P0-1,32*t1-32*t2+31);t4++) {
//         if (t2 == 0) {
//           S0()
//           S1()
//           S2()
//           S3()
//         }
//         for (t5=max(32*t2,-t4+P0+1);t5<=min(P0-1,32*t2+31);t5++) {
//           S0()
//           S1()
//           S2()
//           S3()
//           for (t6=-t4+P0;t6<=min(31,t5-1);t6++) {
//             S4()
//           }
//         }
//       }
//     }
//     if (t1 <= floord(32*t2+P0-63,32)) {
//       for (t4=max(max(1,32*t1-32*t2),-32*t2+P0-31);t4<=32*t1-32*t2+31;t4++) {
//         for (t5=max(32*t2,-t4+P0);t5<=min(P0-1,32*t2+31);t5++) {
//           S0()
//           S1()
//           S2()
//           S3()
//         }
//       }
//     }
//     for (t3=max(1,ceild(-32*t1+32*t2+P0-62,32));t3<=min(floord(P0-2,32),t2);t3++) {
//       for (t4=max(max(max(2,32*t1-32*t2),-32*t2+P0-30),-32*t3+P0-31);t4<=min(P0-1,32*t1-32*t2+31);t4++) {
//         for (t5=max(max(32*t2,32*t3+1),-t4+P0+1);t5<=min(P0-1,32*t2+31);t5++) {
//           for (t6=max(32*t3,-t4+P0);t6<=min(32*t3+31,t5-1);t6++) {
//             S4()
//           }
//         }
//       }
//     }
//   }
// }
