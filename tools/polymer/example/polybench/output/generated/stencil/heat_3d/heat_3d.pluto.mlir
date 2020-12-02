#map0 = affine_map<() -> (1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<() -> (501)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + 1, s1, s2)>
#map4 = affine_map<()[s0, s1, s2] -> (s0, s1, s2)>
#map5 = affine_map<()[s0, s1, s2] -> (s0 - 1, s1, s2)>
#map6 = affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2)>
#map7 = affine_map<()[s0, s1, s2] -> (s0, s1 - 1, s2)>
#map8 = affine_map<()[s0, s1, s2] -> (s0, s1, s2 + 1)>
#map9 = affine_map<()[s0, s1, s2] -> (s0, s1, s2 - 1)>
#map10 = affine_map<(d0) -> (d0 * 8 + 7)>
#map11 = affine_map<()[s0] -> (s0 - 2)>
#map12 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 15)>
#map13 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 13)>
#map14 = affine_map<(d0) -> ((d0 - 1) ceildiv 2)>
#map15 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 12) floordiv 32 + 1)>
#map16 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 2) ceildiv 2)>
#map17 = affine_map<(d0, d1)[s0] -> (d0 * -32 + d1 + s0 - 2)>
#map18 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 - 27)>
#map19 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 - s0 * 2 + 5)>
#map20 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 14)>
#map21 = affine_map<(d0) -> (d0 * 32)>
#map22 = affine_map<(d0)[s0] -> ((d0 * 32 - s0 + 1) ceildiv 2)>
#map23 = affine_map<(d0, d1)[s0] -> (d0 * 32, d1 * 32 - s0 + 3)>
#map24 = affine_map<(d0, d1) -> (d0 * 32 + 1, d1 * 32 + 32)>
#map25 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - s0 + 3, d2 * -32 + d0 * 32 + d1 * 64 - s0 * 2 - 27)>
#map26 = affine_map<(d0) -> (d0 * 32 + 32)>
#map27 = affine_map<(d0) -> (d0 * 8)>
#map28 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 1)>
#map29 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 2)>
#map30 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0)>
#map31 = affine_map<(d0, d1) -> (d0 * -2 + d1)>
#map32 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 1)>
#map33 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0 - 1)>
#map34 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 31)>
#map35 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 - 29)>
#map36 = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
#map37 = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
#map38 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 2 + s0)>
#map39 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 2 + 2, d2 * -32 + d0 * 32 + d1 * 4 - 29)>
#map40 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 4 + 1, d2 * 2 + s0 - 1)>
#map41 = affine_map<(d0, d1, d2) -> (d0 * -32 + d1 * 32 + d2 * 4 + 1)>
#map42 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 2 + s0, d2 * -32 + d0 * 32 + d1 * 4 + 3)>
#map43 = affine_map<(d0, d1, d2, d3)[s0] -> (1, (d0 * 32 - s0 + 2) ceildiv 2, (d1 * 32 - s0 + 2) ceildiv 2, (d2 * 32 - s0 + 2) ceildiv 2, d3 * 8, d3 * 16 - d0 * 16 + 1)>
#map44 = affine_map<(d0, d1, d2, d3)[s0] -> (501, d0 * 16 - d1 * 16 + s0 floordiv 2 + 15, d0 * 8 + 16, d1 * 16 + 15, d2 * 16 + 15, d3 * 16 + 15)>
#map45 = affine_map<(d0, d1)[s0] -> ((d0 * 32 - d1 * 32 + s0 + 29) ceildiv 2)>
#map46 = affine_map<(d0, d1, d2)[s0] -> (d0 * -32 + d1 * 32 + d2 - s0 - 29)>
#map47 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32, d1 * 32 - d2 * 32 + s0 + 30)>
#map48 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 - d2 * 32 + s0 * 2 + 28)>
#map49 = affine_map<(d0) -> (d0 * 16 + 15)>
#map50 = affine_map<(d0, d1) -> (d0 * -32 + d1 - 30)>
#map51 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29)>
#map52 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 + 31, d2 * -32 + d0 * 32 + d1 * 64 + 29)>
#map53 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * 32 + s0 + 29, d2 * -32 + d0 * 32 + d1 * 64 + 61)>
#map54 = affine_map<(d0, d1) -> (d0 * 32, d1 * 32 + 31)>
#map55 = affine_map<(d0) -> (d0 * 8 + 15)>
#map56 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 30)>
#map57 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 29)>
#map58 = affine_map<(d0, d1, d2)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32, (d2 * 32 - s0 - 27) ceildiv 32)>
#map59 = affine_map<(d0, d1, d2)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1, (d2 * 32 + s0 + 27) floordiv 32 + 1)>
#map60 = affine_map<(d0, d1, d2) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + d2 * 64 + 61)>
#map61 = affine_map<(d0, d1) -> (d0 * -16 + d1 * 32)>
#map62 = affine_map<(d0, d1) -> (d0 * -16 + d1 - 31)>
#map63 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 32)>
#map64 = affine_map<(d0) -> (d0 - 1001)>
#map65 = affine_map<(d0) -> (1002, d0 * 32)>
#map66 = affine_map<(d0, d1) -> (1002, d0 * 32, d1 * -32 + d0 * 32 + 1971)>
#map67 = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * -32 + d0 * 32 + 2003)>
#map68 = affine_map<(d0, d1)[s0] -> (0, (d0 - 1) ceildiv 2, (d1 * 32 - s0 - 27) ceildiv 32)>
#map69 = affine_map<(d0, d1)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 28) floordiv 32 + 1, (d1 * 32 + s0 + 27) floordiv 32 + 1)>
#map70 = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 + 31)>
#map71 = affine_map<(d0) -> (d0 ceildiv 2)>
#map72 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 28) floordiv 32 + 1)>
#map73 = affine_map<(d0, d1)[s0] -> (d0 * 32 + 32, d1 * 16 + s0 + 30)>
#map74 = affine_map<(d0) -> ((d0 + 1) ceildiv 2)>
#map75 = affine_map<(d0)[s0] -> ((d0 * 16 + s0 + 29) floordiv 32 + 1)>
#map76 = affine_map<(d0)[s0] -> (s0 + 1000, d0 * 32 + 32)>
#map77 = affine_map<() -> (31)>
#map78 = affine_map<()[s0] -> ((s0 + 999) floordiv 32 + 1)>
#map79 = affine_map<(d0)[s0] -> ((s0 + 998) floordiv 32 + 1, (d0 * 16 + s0 + 13) floordiv 32 + 1, (d0 * 32 + s0 + 26) floordiv 32 + 1)>
#map80 = affine_map<(d0) -> (d0 - 2)>
#map81 = affine_map<(d0) -> (3, d0 * 32)>
#map82 = affine_map<(d0)[s0] -> (s0 + 1, d0 * 32 + 32)>
#map83 = affine_map<() -> (0)>
#map84 = affine_map<()[s0] -> ((s0 - 5) floordiv 32 + 1)>
#map85 = affine_map<() -> (-1)>
#map86 = affine_map<() -> (63)>

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
    %cst = constant 1.250000e-01 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %0 = affine.load %arg4[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %2 = affine.load %arg4[symbol(%arg1) - 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %3 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<120x120x120xf64>
    %4 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %5 = affine.load %arg4[symbol(%arg1), symbol(%arg2) - 1, symbol(%arg3)] : memref<120x120x120xf64>
    %6 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<120x120x120xf64>
    %7 = mulf %cst_0, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst, %9 : f64
    %11 = mulf %cst_0, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %17 = mulf %cst_0, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    return
  }
  func @S1(%arg0: memref<120x120x120xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<120x120x120xf64>) attributes {scop.stmt} {
    %cst = constant 1.250000e-01 : f64
    %cst_0 = constant 2.000000e+00 : f64
    %0 = affine.load %arg4[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %2 = affine.load %arg4[symbol(%arg1) - 1, symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %3 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<120x120x120xf64>
    %4 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %5 = affine.load %arg4[symbol(%arg1), symbol(%arg2) - 1, symbol(%arg3)] : memref<120x120x120xf64>
    %6 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<120x120x120xf64>
    %7 = mulf %cst_0, %1 : f64
    %8 = subf %0, %7 : f64
    %9 = addf %8, %2 : f64
    %10 = mulf %cst, %9 : f64
    %11 = mulf %cst_0, %4 : f64
    %12 = subf %3, %11 : f64
    %13 = addf %12, %5 : f64
    %14 = mulf %cst, %13 : f64
    %15 = addf %10, %14 : f64
    %16 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %17 = mulf %cst_0, %16 : f64
    %18 = subf %6, %17 : f64
    %19 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 1] : memref<120x120x120xf64>
    %20 = addf %18, %19 : f64
    %21 = mulf %cst, %20 : f64
    %22 = addf %15, %21 : f64
    %23 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    %24 = addf %22, %23 : f64
    affine.store %24, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<120x120x120xf64>
    return
  }
  func @kernel_heat_3d_new(%arg0: memref<120x120x120xf64>, %arg1: memref<120x120x120xf64>, %arg2: i32, %arg3: i32) {
    %c1 = constant 1 : index
    %c500 = constant 500 : index
    %0 = index_cast %arg3 : i32 to index
    affine.for %arg4 = -1 to 63 {
      affine.for %arg5 = #map71(%arg4) to min #map79(%arg4)[%0] {
        affine.if #set2(%arg4, %arg5)[%0] {
          affine.if #set1(%arg4) {
            affine.for %arg6 = #map14(%arg4) to #map15(%arg4)[%0] {
              affine.for %arg7 = max #map12(%arg4, %arg6) to min #map13(%arg4, %arg6)[%0] {
                affine.if #set0(%arg4)[%0] {
                  %1 = affine.apply #map10(%arg4)
                  %2 = affine.apply #map11()[%0]
                  call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.for %arg6 = max #map68(%arg4, %arg5)[%0] to min #map69(%arg4, %arg5)[%0] {
          affine.if #set4(%arg4, %arg5, %arg6)[%0] {
            affine.if #set3()[%0] {
              affine.for %arg7 = max #map18(%arg4, %arg5, %arg6)[%0] to min #map19(%arg4, %arg5, %arg6)[%0] {
                %1 = affine.apply #map16(%arg6)[%0]
                %2 = affine.apply #map17(%arg6, %arg7)[%0]
                %3 = affine.apply #map11()[%0]
                call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set2(%arg4, %arg5)[%0] {
            affine.if #set1(%arg4) {
              affine.for %arg7 = #map21(%arg6) to min #map13(%arg4, %arg6)[%0] {
                affine.if #set0(%arg4)[%0] {
                  %1 = affine.apply #map10(%arg4)
                  %2 = affine.apply #map11()[%0]
                  %3 = affine.apply #map20(%arg4, %arg7)
                  call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.for %arg7 = max #map58(%arg4, %arg5, %arg6)[%0] to min #map59(%arg4, %arg5, %arg6)[%0] {
            affine.if #set6(%arg4, %arg5)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map23(%arg5, %arg6)[%0] to min #map24(%arg5, %arg6) {
                  affine.for %arg9 = max #map23(%arg5, %arg7)[%0] to min #map24(%arg5, %arg7) {
                    %1 = affine.apply #map22(%arg5)[%0]
                    %2 = affine.apply #map11()[%0]
                    %3 = affine.apply #map17(%arg5, %arg8)[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set7(%arg4, %arg5, %arg6)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map25(%arg4, %arg5, %arg6)[%0] to min #map19(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map23(%arg6, %arg7)[%0] to min #map24(%arg6, %arg7) {
                    %1 = affine.apply #map22(%arg6)[%0]
                    %2 = affine.apply #map17(%arg6, %arg8)[%0]
                    %3 = affine.apply #map11()[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set8(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map25(%arg4, %arg5, %arg7)[%0] to min #map19(%arg4, %arg5, %arg7)[%0] {
                  affine.for %arg9 = max #map23(%arg6, %arg7)[%0] to #map26(%arg6) {
                    %1 = affine.apply #map22(%arg7)[%0]
                    %2 = affine.apply #map17(%arg7, %arg8)[%0]
                    %3 = affine.apply #map17(%arg7, %arg9)[%0]
                    call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set10(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map29(%arg4, %arg6) to min #map30(%arg4, %arg6)[%0] {
                affine.for %arg9 = max #map29(%arg4, %arg7) to min #map30(%arg4, %arg7)[%0] {
                  affine.if #set9(%arg4) {
                    %1 = affine.apply #map27(%arg4)
                    %2 = affine.apply #map28(%arg4, %arg8)
                    call @S1(%arg0, %1, %c1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.for %arg8 = max #map43(%arg4, %arg5, %arg6, %arg7)[%0] to min #map44(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set11(%arg4, %arg5, %arg8) {
                affine.for %arg9 = max #map32(%arg6, %arg8) to min #map33(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map32(%arg7, %arg8) to min #map33(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map31(%arg8, %arg9)
                    call @S0(%arg1, %arg8, %c1, %1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map34(%arg4, %arg5, %arg8) to #map35(%arg4, %arg5, %arg8) {
                affine.for %arg10 = max #map32(%arg6, %arg8) to min #map33(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map32(%arg7, %arg8) to min #map33(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map31(%arg8, %arg9)
                    %2 = affine.apply #map31(%arg8, %arg10)
                    call @S0(%arg1, %arg8, %1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = max #map39(%arg4, %arg5, %arg8) to min #map40(%arg4, %arg5, %arg8)[%0] {
                affine.if #set12(%arg6, %arg8) {
                  affine.for %arg10 = max #map32(%arg7, %arg8) to min #map33(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map31(%arg8, %arg9)
                    call @S0(%arg1, %arg8, %1, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.for %arg10 = max #map37(%arg6, %arg8) to min #map33(%arg6, %arg8)[%0] {
                  affine.if #set12(%arg7, %arg8) {
                    %1 = affine.apply #map31(%arg8, %arg9)
                    %2 = affine.apply #map31(%arg8, %arg10)
                    call @S0(%arg1, %arg8, %1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.for %arg11 = max #map37(%arg7, %arg8) to min #map33(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map36(%arg8, %arg9)
                    %2 = affine.apply #map36(%arg8, %arg10)
                    call @S1(%arg0, %arg8, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                    %3 = affine.apply #map31(%arg8, %arg9)
                    %4 = affine.apply #map31(%arg8, %arg10)
                    call @S0(%arg1, %arg8, %3, %4, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                  affine.if #set13(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map36(%arg8, %arg9)
                    %2 = affine.apply #map36(%arg8, %arg10)
                    call @S1(%arg0, %arg8, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
                affine.if #set13(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map37(%arg7, %arg8) to min #map38(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map36(%arg8, %arg9)
                    %2 = affine.apply #map11()[%0]
                    call @S1(%arg0, %arg8, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.for %arg9 = #map41(%arg4, %arg5, %arg8) to min #map42(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg10 = max #map37(%arg6, %arg8) to min #map38(%arg6, %arg8)[%0] {
                  affine.for %arg11 = max #map37(%arg7, %arg8) to min #map38(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map36(%arg8, %arg9)
                    %2 = affine.apply #map36(%arg8, %arg10)
                    call @S1(%arg0, %arg8, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
              affine.if #set14(%arg4, %arg5, %arg8)[%0] {
                affine.for %arg9 = max #map37(%arg6, %arg8) to min #map38(%arg6, %arg8)[%0] {
                  affine.for %arg10 = max #map37(%arg7, %arg8) to min #map38(%arg7, %arg8)[%0] {
                    %1 = affine.apply #map11()[%0]
                    %2 = affine.apply #map36(%arg8, %arg9)
                    call @S1(%arg0, %arg8, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set15(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.if #set5()[%0] {
                affine.for %arg8 = max #map47(%arg4, %arg5, %arg6)[%0] to min #map48(%arg4, %arg5, %arg6)[%0] {
                  affine.for %arg9 = max #map47(%arg4, %arg5, %arg7)[%0] to min #map48(%arg4, %arg5, %arg7)[%0] {
                    %1 = affine.apply #map45(%arg4, %arg5)[%0]
                    %2 = affine.apply #map11()[%0]
                    %3 = affine.apply #map46(%arg4, %arg5, %arg8)[%0]
                    call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
            affine.if #set16(%arg4, %arg5, %arg6, %arg7)[%0] {
              affine.for %arg8 = max #map52(%arg4, %arg5, %arg6) to min #map53(%arg4, %arg5, %arg6)[%0] {
                affine.for %arg9 = #map21(%arg7) to min #map51(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map49(%arg6)
                  %2 = affine.apply #map50(%arg6, %arg8)
                  call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set17(%arg4, %arg5, %arg7)[%0] {
              affine.for %arg8 = max #map52(%arg4, %arg5, %arg7) to min #map53(%arg4, %arg5, %arg7)[%0] {
                affine.for %arg9 = max #map54(%arg6, %arg7) to min #map51(%arg6, %arg7)[%0] {
                  %1 = affine.apply #map49(%arg7)
                  %2 = affine.apply #map50(%arg7, %arg8)
                  %3 = affine.apply #map50(%arg7, %arg9)
                  call @S0(%arg1, %1, %2, %3, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
            affine.if #set18(%arg4, %arg5, %arg6, %arg7) {
              affine.for %arg8 = #map21(%arg6) to min #map57(%arg4, %arg6)[%0] {
                affine.for %arg9 = #map21(%arg7) to min #map57(%arg4, %arg7)[%0] {
                  affine.if #set9(%arg4) {
                    %1 = affine.apply #map55(%arg4)
                    %2 = affine.apply #map56(%arg4, %arg8)
                    call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                  }
                }
              }
            }
          }
          affine.if #set20(%arg4, %arg6) {
            affine.if #set19()[%0] {
              affine.for %arg7 = max #map52(%arg4, %arg5, %arg6) to min #map60(%arg4, %arg5, %arg6) {
                %1 = affine.apply #map49(%arg6)
                %2 = affine.apply #map50(%arg6, %arg7)
                call @S0(%arg1, %1, %2, %c1, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set22(%arg4, %arg5, %arg6) {
            affine.if #set21(%arg4)[%0] {
              affine.for %arg7 = #map21(%arg6) to #map26(%arg6) {
                affine.if #set9(%arg4) {
                  %1 = affine.apply #map55(%arg4)
                  %2 = affine.apply #map56(%arg4, %arg7)
                  call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
          affine.if #set24(%arg4, %arg5, %arg6) {
            affine.if #set23(%arg4)[%0] {
              affine.for %arg7 = max #map63(%arg4, %arg6) to #map26(%arg6) {
                %1 = affine.apply #map55(%arg4)
                %2 = affine.apply #map61(%arg4, %arg5)
                %3 = affine.apply #map62(%arg4, %arg7)
                call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
          affine.if #set26(%arg4, %arg6) {
            affine.if #set25()[%0] {
              affine.for %arg7 = max #map66(%arg4, %arg5) to min #map67(%arg4, %arg5) {
                affine.for %arg8 = max #map65(%arg6) to #map26(%arg6) {
                  %1 = affine.apply #map64(%arg7)
                  %2 = affine.apply #map64(%arg8)
                  call @S1(%arg0, %c500, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set27(%arg4, %arg5) {
          affine.if #set21(%arg4)[%0] {
            affine.for %arg6 = #map71(%arg4) to #map72(%arg4)[%0] {
              affine.for %arg7 = max #map70(%arg4, %arg6) to min #map57(%arg4, %arg6)[%0] {
                affine.if #set9(%arg4) {
                  %1 = affine.apply #map55(%arg4)
                  %2 = affine.apply #map11()[%0]
                  call @S0(%arg1, %1, %c1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
        affine.if #set28(%arg4, %arg5) {
          affine.if #set23(%arg4)[%0] {
            affine.for %arg6 = #map74(%arg4) to #map75(%arg4)[%0] {
              affine.for %arg7 = max #map63(%arg4, %arg6) to min #map73(%arg4, %arg6)[%0] {
                %1 = affine.apply #map55(%arg4)
                %2 = affine.apply #map61(%arg4, %arg5)
                %3 = affine.apply #map11()[%0]
                call @S1(%arg0, %1, %2, %3, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
              }
            }
          }
        }
        affine.if #set29(%arg4) {
          affine.if #set25()[%0] {
            affine.for %arg6 = 31 to #map78()[%0] {
              affine.for %arg7 = max #map66(%arg4, %arg5) to min #map67(%arg4, %arg5) {
                affine.for %arg8 = max #map65(%arg6) to min #map76(%arg6)[%0] {
                  %1 = affine.apply #map64(%arg7)
                  %2 = affine.apply #map11()[%0]
                  call @S1(%arg0, %c500, %1, %2, %arg1) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set31(%arg4) {
        affine.if #set30()[%0] {
          affine.for %arg5 = 0 to #map84()[%0] {
            affine.for %arg6 = 0 to #map84()[%0] {
              affine.for %arg7 = max #map81(%arg5) to min #map82(%arg5)[%0] {
                affine.for %arg8 = max #map81(%arg6) to min #map82(%arg6)[%0] {
                  %1 = affine.apply #map11()[%0]
                  %2 = affine.apply #map80(%arg7)
                  call @S0(%arg1, %c1, %1, %2, %arg0) : (memref<120x120x120xf64>, index, index, index, memref<120x120x120xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.if #set32(%arg4) {
        affine.if #set25()[%0] {
          affine.for %arg5 = 31 to #map78()[%0] {
            affine.for %arg6 = 31 to #map78()[%0] {
              affine.for %arg7 = max #map65(%arg5) to min #map76(%arg5)[%0] {
                affine.for %arg8 = max #map65(%arg6) to min #map76(%arg6)[%0] {
                  %1 = affine.apply #map11()[%0]
                  %2 = affine.apply #map64(%arg7)
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
