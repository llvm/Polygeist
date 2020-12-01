#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (501)>
#map3 = affine_map<(d0) -> (d0 + 1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0) -> (d0 - 1)>
#map6 = affine_map<(d0) -> (d0 * 8 + 7)>
#map7 = affine_map<()[s0] -> (s0 - 2)>
#map8 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 15)>
#map9 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 13)>
#map10 = affine_map<(d0) -> ((d0 - 1) ceildiv 2)>
#map11 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 12) floordiv 32 + 1)>
#map12 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 2) ceildiv 2)>
#map13 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map14 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 - 27)>
#map15 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 + 5)>
#map16 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 14)>
#map17 = affine_map<(d0) -> (d0 * 32)>
#map18 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 1) ceildiv 2)>
#map19 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map20 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map21 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d0 * 32 + d1 * 64 - s0 * 2 - 27)>
#map22 = affine_map<(d0) -> (d0 * 32 + 32)>
#map23 = affine_map<(d0) -> (d0 * 8)>
#map24 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 1)>
#map25 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 2)>
#map26 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0)>
#map27 = affine_map<(d0) -> (d0)>
#map28 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map29 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map30 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map31 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 31)>
#map32 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 - 29)>
#map33 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map34 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map35 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map36 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 29)>
#map37 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + s0 - 1)>
#map38 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 + 1)>
#map39 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 2 + s0, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map40 = affine_map<(d0, d1, d2, d3)[s0] -> (1, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, (d2 * 32 - s0 + 2) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map41 = affine_map<(d0, d1, d2, d3)[s0] -> (501, d0 * 16 - d1 * 16 + s0 floordiv 2 + 15, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map42 = affine_map<(d0, d1)[s0] -> ((d0 * 32 - d1 * 32 + s0 + 29) ceildiv 2)>
#map43 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 32 + d2 - s0 - 29)>
#map44 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - d2 * 32 + s0 + 30)>
#map45 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + s0 * 2 + 28)>
#map46 = affine_map<(d0) -> (d0 * 16 + 15)>
#map47 = affine_map<(d0, d1) -> (d0 * -32 + d1 - 30)>
#map48 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29)>
#map49 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 + 31, d2 * -32 + d0 * 32 + d1 * 64 + 29)>
#map50 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29, d2 * -32 + d0 * 32 + d1 * 64 + 61)>
#map51 = affine_map<(d0, d1) -> (d0 * 32, d1 * 32 + 31)>
#map52 = affine_map<(d0) -> (d0 * 8 + 15)>
#map53 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 30)>
#map54 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 29)>
#map55 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32, (d2 * 32 - s0 - 27) ceildiv 32)>
#map56 = affine_map<(d0, d1, d2)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1, (d2 * 32 + s0 + 27) floordiv 32 + 1)>
#map57 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 + 61)>
#map58 = affine_map<(d0, d1) -> (d0 * -16 + d1 * 32)>
#map59 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 31)>
#map60 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 32)>
#map61 = affine_map<(d0) -> (d0 - 1001)>
#map62 = affine_map<(d0) -> (1002, d0 * 32)>
#map63 = affine_map<(d0, d1) -> (1002, d0 * 32, d1 * -32 + d0 * 32 + 1971)>
#map64 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 2003)>
#map65 = affine_map<(d0, d1)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32)>
#map66 = affine_map<(d0, d1)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1)>
#map67 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 31)>
#map68 = affine_map<(d0) -> (d0 ceildiv 2)>
#map69 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 28) floordiv 32 + 1)>
#map70 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 30)>
#map71 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map72 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 29) floordiv 32 + 1)>
#map73 = affine_map<(d0)[s0] -> (s0 + 1000, d0 * 32 + 32)>
#map74 = affine_map<() -> (31)>
#map75 = affine_map<()[s0] -> ((s0 + 999) floordiv 32 + 1)>
#map76 = affine_map<(d0)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 13) floordiv 32 + 1, (d0 * 32 + s0 + 26) floordiv 32 + 1)>
#map77 = affine_map<(d0) -> (d0 - 2)>
#map78 = affine_map<(d0) -> (3, d0 * 32)>
#map79 = affine_map<(d0)[s0] -> (s0 + 1, d0 * 32 + 32)>
#map80 = affine_map<() -> (0)>
#map81 = affine_map<()[s0] -> ((s0 - 5) floordiv 32 + 1)>
#map82 = affine_map<() -> (-1)>
#map83 = affine_map<() -> (63)>

#set0 = affine_set<(d0)[s0] : ((d0 * 16 + s0 * 31 + 20) mod 32 == 0)>
#set1 = affine_set<(d0) : ((d0 + 1) mod 2 == 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 * 16 - (d1 * 32 - s0 - 12) == 0)>
#set3 = affine_set<()[s0] : ((s0 * 31 + 4) mod 32 == 0)>
#set4 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 - 12) floordiv 16 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 28) ceildiv 32 >= 0)>
#set5 = affine_set<()[s0] : ((s0 + 1) mod 2 == 0)>
#set6 = affine_set<(d0, d1)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set7 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set8 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d2 * 32 + d1 * 32 - s0 - 1) floordiv 32 - d0 >= 0, d1 - d2 - 1 >= 0, d1 - d3 - 1 >= 0, d1 - (s0 + 1) ceildiv 32 >= 0)>
#set9 = affine_set<(d0) : (d0 mod 2 == 0)>
#set10 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - d1 * 2 == 0, d0 - 2 >= 0, d0 - (d2 * 32 - s0 + 1) ceildiv 16 >= 0, d0 - (d3 * 32 - s0 + 1) ceildiv 16 >= 0)>
#set11 = affine_set<(d0, d1, d2) : (d0 - (d1 * 16 + d2 - 16) ceildiv 16 >= 0, d2 floordiv 16 - d1 >= 0)>
#set12 = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
#set13 = affine_set<(d0, d1)[s0] : (d0 - (d1 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set14 = affine_set<(d0, d1, d2)[s0] : ((d1 * 32 + d2 * 2 - s0 + 1) floordiv 32 - d0 >= 0, d1 - (d2 * 2 + s0 - 32) ceildiv 32 >= 0)>
#set15 = affine_set<(d0, d1, d2, d3)[s0] : ((d1 * 32 - s0 + 1) floordiv 16 - d0 >= 0, (d1 * 32 - s0 + 971) floordiv 32 - d0 >= 0, (d1 * 32 + d2 * 32 - s0 - 1) floordiv 32 - d0 >= 0, (d1 * 32 + d3 * 32 - s0 - 1) floordiv 32 - d0 >= 0)>
#set16 = affine_set<(d0, d1, d2, d3)[s0] : (d0 - (d1 * 32 + d2 * 32 - s0 + 1) ceildiv 32 >= 0, d0 - d2 * 2 >= 0, -d2 + 30 >= 0, d3 - d2 - 1 >= 0)>
#set17 = affine_set<(d0, d1, d2)[s0] : (d0 - (d1 * 32 + d2 * 32 - s0 + 1) ceildiv 32 >= 0, d0 - d2 * 2 >= 0, -d2 + 30 >= 0)>
#set18 = affine_set<(d0, d1, d2, d3) : (d0 - d1 * 2 == 0, -d0 + 60 >= 0, d2 * 2 - d0 - 2 >= 0, d3 * 2 - d0 - 2 >= 0)>
#set19 = affine_set<()[s0] : ((s0 + 28) mod 32 == 0)>
#set20 = affine_set<(d0, d1) : (d0 - d1 * 2 >= 0, -d1 + 30 >= 0)>
#set21 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 28) mod 32 == 0)>
#set22 = affine_set<(d0, d1, d2) : (d0 - d1 * 2 == 0, -d0 + 60 >= 0, d2 * 2 - d0 - 2 >= 0)>
#set23 = affine_set<(d0)[s0] : ((d0 * 16 + s0 + 29) mod 32 == 0)>
#set24 = affine_set<(d0, d1, d2) : (-d0 + 60 >= 0, d1 * 2 - d0 - 1 >= 0, d2 * 2 - d0 - 1 >= 0)>
#set25 = affine_set<()[s0] : ((s0 + 7) mod 32 == 0)>
#set26 = affine_set<(d0, d1) : (d0 - 61 >= 0, d1 - 31 >= 0)>
#set27 = affine_set<(d0, d1) : (-d0 + 60 >= 0, d0 - d1 * 2 == 0)>
#set28 = affine_set<(d0, d1) : (-d0 + 60 >= 0, d1 * 2 - d0 - 1 >= 0)>
#set29 = affine_set<(d0) : (d0 - 61 >= 0)>
#set30 = affine_set<()[s0] : ((s0 + 27) mod 32 == 0)>
#set31 = affine_set<(d0) : (d0 + 1 == 0)>
#set32 = affine_set<(d0) : (d0 - 62 == 0)>

module {
  func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to #map1()[%0] {
        affine.for %arg6 = 1 to #map1()[%0] {
          affine.for %arg7 = 1 to #map1()[%0] {
            call @S0(%arg3, %arg5, %arg6, %arg7, %arg2) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
        }
      }
      affine.for %arg5 = 1 to #map1()[%0] {
        affine.for %arg6 = 1 to #map1()[%0] {
          affine.for %arg7 = 1 to #map1()[%0] {
            call @S1(%arg2, %arg5, %arg6, %arg7, %arg3) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
          }
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map3(%arg1)
    %1 = affine.load %arg4[%0, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.apply #map5(%arg1)
    %4 = affine.load %arg4[%3, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.apply #map3(%arg2)
    %6 = affine.load %arg4[%arg1, %5, %arg3] : memref<120x120x120xf64>
    %7 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %8 = affine.apply #map5(%arg2)
    %9 = affine.load %arg4[%arg1, %8, %arg3] : memref<120x120x120xf64>
    %cst = constant 1.250000e-01 : f64
    %10 = affine.apply #map3(%arg3)
    %11 = affine.load %arg4[%arg1, %arg2, %10] : memref<120x120x120xf64>
    %cst_0 = constant 2.000000e+00 : f64
    %12 = mulf %cst_0, %2 : f64
    %13 = subf %1, %12 : f64
    %14 = addf %13, %4 : f64
    %15 = mulf %cst, %14 : f64
    %16 = mulf %cst_0, %7 : f64
    %17 = subf %6, %16 : f64
    %18 = addf %17, %9 : f64
    %19 = mulf %cst, %18 : f64
    %20 = addf %15, %19 : f64
    %21 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %22 = mulf %cst_0, %21 : f64
    %23 = subf %11, %22 : f64
    %24 = affine.apply #map5(%arg3)
    %25 = affine.load %arg4[%arg1, %arg2, %24] : memref<120x120x120xf64>
    %26 = addf %23, %25 : f64
    %27 = mulf %cst, %26 : f64
    %28 = addf %20, %27 : f64
    %29 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %30 = addf %28, %29 : f64
    affine.store %30, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
  func @S1(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %0 = affine.apply #map3(%arg1)
    %1 = affine.load %arg4[%0, %arg2, %arg3] : memref<120x120x120xf64>
    %2 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %3 = affine.apply #map5(%arg1)
    %4 = affine.load %arg4[%3, %arg2, %arg3] : memref<120x120x120xf64>
    %5 = affine.apply #map3(%arg2)
    %6 = affine.load %arg4[%arg1, %5, %arg3] : memref<120x120x120xf64>
    %7 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %8 = affine.apply #map5(%arg2)
    %9 = affine.load %arg4[%arg1, %8, %arg3] : memref<120x120x120xf64>
    %cst = constant 1.250000e-01 : f64
    %10 = affine.apply #map3(%arg3)
    %11 = affine.load %arg4[%arg1, %arg2, %10] : memref<120x120x120xf64>
    %cst_0 = constant 2.000000e+00 : f64
    %12 = mulf %cst_0, %2 : f64
    %13 = subf %1, %12 : f64
    %14 = addf %13, %4 : f64
    %15 = mulf %cst, %14 : f64
    %16 = mulf %cst_0, %7 : f64
    %17 = subf %6, %16 : f64
    %18 = addf %17, %9 : f64
    %19 = mulf %cst, %18 : f64
    %20 = addf %15, %19 : f64
    %21 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %22 = mulf %cst_0, %21 : f64
    %23 = subf %11, %22 : f64
    %24 = affine.apply #map5(%arg3)
    %25 = affine.load %arg4[%arg1, %arg2, %24] : memref<120x120x120xf64>
    %26 = addf %23, %25 : f64
    %27 = mulf %cst, %26 : f64
    %28 = addf %20, %27 : f64
    %29 = affine.load %arg4[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    %30 = addf %28, %29 : f64
    affine.store %30, %arg0[%arg1, %arg2, %arg3] : memref<120x120x120xf64>
    return
  }
  func @kernel_heat_3d_new(%arg0: memref<120x120x120xf64>, %arg1: memref<120x120x120xf64>, %arg2: i32, %arg3: i32) {
    %0 = index_cast %arg3 : i32 to index
    affine.for %arg4 = -1 to 63 {
      affine.for %arg5 = #map68(%arg4) to min #map76(%arg4)[%0] {
        affine.if #set2(%arg4, %arg5)[%0] {
          affine.if #set1(%arg4) {
            affine.for %arg6 = #map10(%arg4) to #map11(%arg4)[%0] {
              affine.for %arg7 = max #map8(%arg4, %arg6) to min #map9(%arg4, %arg6)[%0] {
                affine.if #set0(%arg4)[%0] {
                  %1 = affine.apply #map6(%arg4)
                  %2 = affine.apply #map7()[%0]
                  %c1 = constant 1 : index
                  call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.for %arg6 = max #map65(%arg4, %arg5)[%0] to min #map66(%arg4, %arg5)[%0] {
          affine.if #set4(%arg4, %arg5, %arg6)[%0] {
            affine.if #set3()[%0] {
              affine.for %arg7 = max #map14(%arg4, %arg5, %arg6)[%0] to min #map15(%arg4, %arg5, %arg6)[%0] {
                %1 = affine.apply #map12(%arg6)[%0]
                %2 = affine.apply #map13(%arg6, %arg7)[%0]
                %3 = affine.apply #map7()[%0]
                call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set2(%arg4, %arg5)[%0] {
            affine.if #set1(%arg4) {
              affine.for %arg7 = #map17(%arg6) to min #map9(%arg4, %arg6)[%0] {
                affine.if #set0(%arg4)[%0] {
                  %1 = affine.apply #map6(%arg4)
                  %2 = affine.apply #map7()[%0]
                  %3 = affine.apply #map16(%arg4, %arg7)
                  call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.for %arg7 = max #map55(%arg4, %arg5, %arg6)[%0] to min #map56(%arg4, %arg5, %arg6)[%0] {
            affine.if #set6(%arg4, %arg5)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map19(%arg5, %arg6)[%0] to min #map20(%arg5, %arg6) {
                  affine.for %arg9 = max #map19(%arg5, %arg7)[%0] to min #map20(%arg5, %arg7) {
                    %1 = affine.apply #map18(%arg5)[%0]
                    %2 = affine.apply #map7()[%0]
                    %3 = affine.apply #map13(%arg5, %arg8)[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set7(%arg4, %arg5, %arg6)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map21(%arg4, %arg5, %arg6)[%0] to min #map15(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map19(%arg6, %arg7)[%0] to min #map20(%arg6, %arg7) {
                    %1 = affine.apply #map18(%arg6)[%0]
                    %2 = affine.apply #map13(%arg6, %arg8)[%0]
                    %3 = affine.apply #map7()[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set8(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map21(%arg4, %arg5, %arg7)[%0] to min #map15(%arg4, %arg5, %arg7)[%0] {
                  affine.for %arg9 = max #map19(%arg6, %arg7)[%0] to #map22(%arg6) {
                    %1 = affine.apply #map18(%arg7)[%0]
                    %2 = affine.apply #map13(%arg7, %arg8)[%0]
                    %3 = affine.apply #map13(%arg7, %arg9)[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set10(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map25(%arg4, %arg6) to min #map26(%arg4, %arg6)[%0] {
                affine.for %arg9 = max #map25(%arg4, %arg7) to min #map26(%arg4, %arg7)[%0] {
                  affine.if #set9(%arg4) {
                    %1 = affine.apply #map23(%arg4)
                    %c1 = constant 1 : index
                    %2 = affine.apply #map24(%arg4, %arg8)
                    call @S1(%arg0, %1, %c1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.for %arg8 = max #map40(%arg4, %arg5, %arg6, %arg7)[%0] to min #map41(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set11(%arg4, %arg5, %arg8) {
                affine.for %arg9 = max #map29(%arg6, %arg8) to min #map30(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map29(%arg7, %arg8) to min #map30(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %c1 = constant 1 : index
                    %2 = affine.apply #map28(%arg8, %arg9)
                    call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map31(%arg4, %arg5, %arg8) to #map32(%arg4, %arg5, %arg8) {
                affine.for %arg10 = max #map29(%arg6, %arg8) to min #map30(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map29(%arg7, %arg8) to min #map30(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map28(%arg8, %arg9)
                    %3 = affine.apply #map28(%arg8, %arg10)
                    call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map36(%arg4, %arg5, %arg8) to min #map37(%arg4, %arg5, %arg8)[%0] {
                affine.if #set12(%arg6, %arg8) {
                  affine.for %arg10 = max #map29(%arg7, %arg8) to min #map30(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map28(%arg8, %arg9)
                    %c1 = constant 1 : index
                    call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.for %arg10 = max #map34(%arg6, %arg8) to min #map30(%arg6, %arg8)[%0] {
                  affine.if #set12(%arg7, %arg8) {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map28(%arg8, %arg9)
                    %3 = affine.apply #map28(%arg8, %arg10)
                    call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.for %arg11 = max #map34(%arg7, %arg8) to min #map30(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map33(%arg8, %arg9)
                    %3 = affine.apply #map33(%arg8, %arg10)
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                    %4 = affine.apply #map28(%arg8, %arg9)
                    %5 = affine.apply #map28(%arg8, %arg10)
                    call @S0(%arg1, %1, %4, %5, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.if #set13(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map33(%arg8, %arg9)
                    %3 = affine.apply #map33(%arg8, %arg10)
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.if #set13(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map34(%arg7, %arg8) to min #map35(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map33(%arg8, %arg9)
                    %3 = affine.apply #map7()[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = #map38(%arg4, %arg5, %arg8) to min #map39(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg10 = max #map34(%arg6, %arg8) to min #map35(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map34(%arg7, %arg8) to min #map35(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map33(%arg8, %arg9)
                    %3 = affine.apply #map33(%arg8, %arg10)
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.if #set14(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg9 = max #map34(%arg6, %arg8) to min #map35(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map34(%arg7, %arg8) to min #map35(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map27(%arg8)
                    %2 = affine.apply #map7()[%0]
                    %3 = affine.apply #map33(%arg8, %arg9)
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set15(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map44(%arg4, %arg5, %arg6)[%0] to min #map45(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map44(%arg4, %arg5, %arg7)[%0] to min #map45(%arg4, %arg5, %arg7)[%0] {
                    %1 = affine.apply #map42(%arg4, %arg5)[%0]
                    %2 = affine.apply #map7()[%0]
                    %3 = affine.apply #map43(%arg4, %arg5, %arg8)[%0]
                    call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set16(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map49(%arg4, %arg5, %arg6) to min #map50(%arg4, %arg5, %arg6)[%0] {
                affine.for %arg9 = #map17(%arg7) to min #map48(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map46(%arg6)
                  %2 = affine.apply #map47(%arg6, %arg8)
                  %c1 = constant 1 : index
                  call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set17(%arg4, %arg5, %arg7)[%0] {
              affine.for %arg8 = max #map49(%arg4, %arg5, %arg7) to min #map50(%arg4, %arg5, %arg7)[%0] {
                affine.for %arg9 = max #map51(%arg6, %arg7) to min #map48(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map46(%arg7)
                  %2 = affine.apply #map47(%arg7, %arg8)
                  %3 = affine.apply #map47(%arg7, %arg9)
                  call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set18(%arg4, %arg5, %arg6, %arg7) {
              affine.for %arg8 = #map17(%arg6) to min #map54(%arg4, %arg6)[%0] {
                affine.for %arg9 = #map17(%arg7) to min #map54(%arg4, %arg7)[%0] {
                  affine.if #set9(%arg4) {
                    %1 = affine.apply #map52(%arg4)
                    %c1 = constant 1 : index
                    %2 = affine.apply #map53(%arg4, %arg8)
                    call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
          }
          affine.if #set20(%arg4, %arg6) {
            affine.if #set19()[%0] {
              affine.for %arg7 = max #map49(%arg4, %arg5, %arg6) to min #map57(%arg4, %arg5, %arg6) {
                %1 = affine.apply #map46(%arg6)
                %2 = affine.apply #map47(%arg6, %arg7)
                %c1 = constant 1 : index
                call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set22(%arg4, %arg5, %arg6) {
            affine.if #set21(%arg4)[%0] {
              affine.for %arg7 = #map17(%arg6) to #map22(%arg6) {
                affine.if #set9(%arg4) {
                  %1 = affine.apply #map52(%arg4)
                  %c1 = constant 1 : index
                  %2 = affine.apply #map53(%arg4, %arg7)
                  call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.if #set24(%arg4, %arg5, %arg6) {
            affine.if #set23(%arg4)[%0] {
              affine.for %arg7 = max #map60(%arg4, %arg6) to #map22(%arg6) {
                %1 = affine.apply #map52(%arg4)
                %2 = affine.apply #map58(%arg4, %arg5)
                %3 = affine.apply #map59(%arg4, %arg7)
                call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set26(%arg4, %arg6) {
            affine.if #set25()[%0] {
              affine.for %arg7 = max #map63(%arg4, %arg5) to min #map64(%arg4, %arg5) {
                affine.for %arg8 = max #map62(%arg6) to #map22(%arg6) {
                  %c500 = constant 500 : index
                  %1 = affine.apply #map61(%arg7)
                  %2 = affine.apply #map61(%arg8)
                  call @S1(%arg0, %c500, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set27(%arg4, %arg5) {
          affine.if #set21(%arg4)[%0] {
            affine.for %arg6 = #map68(%arg4) to #map69(%arg4)[%0] {
              affine.for %arg7 = max #map67(%arg4, %arg6) to min #map54(%arg4, %arg6)[%0] {
                affine.if #set9(%arg4) {
                  %1 = affine.apply #map52(%arg4)
                  %c1 = constant 1 : index
                  %2 = affine.apply #map7()[%0]
                  call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set28(%arg4, %arg5) {
          affine.if #set23(%arg4)[%0] {
            affine.for %arg6 = #map71(%arg4) to #map72(%arg4)[%0] {
              affine.for %arg7 = max #map60(%arg4, %arg6) to min #map70(%arg4, %arg6)[%0] {
                %1 = affine.apply #map52(%arg4)
                %2 = affine.apply #map58(%arg4, %arg5)
                %3 = affine.apply #map7()[%0]
                call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
        }
        affine.if #set29(%arg4) {
          affine.if #set25()[%0] {
            affine.for %arg6 = 31 to #map75()[%0] {
              affine.for %arg7 = max #map63(%arg4, %arg5) to min #map64(%arg4, %arg5) {
                affine.for %arg8 = max #map62(%arg6) to min #map73(%arg6)[%0] {
                  %c500 = constant 500 : index
                  %1 = affine.apply #map61(%arg7)
                  %2 = affine.apply #map7()[%0]
                  call @S1(%arg0, %c500, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set31(%arg4) {
        affine.if #set30()[%0] {
          affine.for %arg5 = 0 to #map81()[%0] {
            affine.for %arg6 = 0 to #map81()[%0] {
              affine.for %arg7 = max #map78(%arg5) to min #map79(%arg5)[%0] {
                affine.for %arg8 = max #map78(%arg6) to min #map79(%arg6)[%0] {
                  %c1 = constant 1 : index
                  %1 = affine.apply #map7()[%0]
                  %2 = affine.apply #map77(%arg7)
                  call @S0(%arg1, %c1, %1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set32(%arg4) {
        affine.if #set25()[%0] {
          affine.for %arg5 = 31 to #map75()[%0] {
            affine.for %arg6 = 31 to #map75()[%0] {
              affine.for %arg7 = max #map62(%arg5) to min #map73(%arg5)[%0] {
                affine.for %arg8 = max #map62(%arg6) to min #map73(%arg6)[%0] {
                  %c500 = constant 500 : index
                  %1 = affine.apply #map7()[%0]
                  %2 = affine.apply #map61(%arg7)
                  call @S1(%arg0, %c500, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
