#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 - 1)>
#map4 = affine_map<(d0)[s0] -> (s0, d0)>
#map5 = affine_map<(d0) -> (d0 - 2)>
#map6 = affine_map<(d0, d1)[s0] -> (s0, d0, d1)>
#map7 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 125 + d0 * 25 + d1 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_divdiv_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c108_i32 = arith.constant 108 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<4x4xf64>
    %alloca_1 = memref.alloca() : memref<5xf64>
    %alloca_2 = memref.alloca() : memref<5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = arith.index_cast %arg7 : index to i32
      %1 = polygeist.submap(%alloca_3, %c5, %c5, %c5) {map = #map} : (memref<5x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%1 : memref<?x?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %2 = arith.muli %0, %c108_i32 : i32
      %alloca_4 = memref.alloca() : memref<i32>
      affine.store %c0_i32, %alloca_4[] : memref<i32>
      affine.for %arg8 = 0 to 3 {
        %6 = affine.load %alloca_4[] : memref<i32>
        %7 = arith.index_cast %arg8 : index to i32
        %8 = arith.cmpi eq, %7, %c2_i32 : i32
        %9 = arith.select %8, %c4_i32, %c3_i32 : i32
        %10 = arith.cmpi eq, %7, %c1_i32 : i32
        %11 = arith.select %10, %c4_i32, %c3_i32 : i32
        %12 = arith.cmpi eq, %7, %c0_i32 : i32
        %13 = arith.select %12, %c4_i32, %c3_i32 : i32
        %14 = arith.index_cast %9 : i32 to index
        %15 = arith.index_cast %11 : i32 to index
        %16 = arith.index_cast %13 : i32 to index
        scf.for %arg9 = %c0 to %14 step %c1 {
          %20 = arith.index_cast %arg9 : index to i32
          %21 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map1} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%21 : memref<?x?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %22 = arith.muli %20, %11 : i32
          scf.for %arg10 = %c0 to %15 step %c1 {
            %23 = arith.index_cast %arg10 : index to i32
            %24 = polygeist.submap(%alloca_1, %c5) {map = #map2} : (memref<5xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%24 : memref<?xf64>) {
            ^bb0(%out: f64):
              linalg.yield %cst : f64
            }
            %25 = arith.addi %23, %22 : i32
            %26 = arith.muli %25, %13 : i32
            scf.for %arg11 = %c0 to %16 step %c1 {
              %27 = arith.index_cast %arg11 : index to i32
              %28 = arith.addi %27, %26 : i32
              %29 = arith.addi %28, %6 : i32
              %30 = arith.addi %29, %2 : i32
              %31 = arith.index_cast %30 : i32 to index
              %32 = memref.load %arg5[%31] : memref<?xf64>
              %33 = polygeist.submap(%alloca_1, %c5) {map = #map2} : (memref<5xf64>, index) -> memref<?xf64>
              linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%33 : memref<?xf64>) {
              ^bb0(%out: f64):
                %34 = linalg.index 0 : index
                %35 = arith.index_cast %34 : index to i32
                %36 = arith.cmpi eq, %arg8, %c0 : index
                %37 = arith.muli %35, %c4_i32 : i32
                %38 = arith.addi %37, %27 : i32
                %39 = arith.index_cast %38 : i32 to index
                %40 = memref.load %arg2[%39] : memref<?xf64>
                %41 = arith.muli %35, %c3_i32 : i32
                %42 = arith.addi %41, %27 : i32
                %43 = arith.index_cast %42 : i32 to index
                %44 = memref.load %arg0[%43] : memref<?xf64>
                %45 = arith.select %36, %40, %44 : f64
                %46 = arith.mulf %32, %45 : f64
                %47 = arith.addf %out, %46 : f64
                linalg.yield %47 : f64
              }
            }
            affine.for %arg11 = 0 to 5 {
              %27 = arith.index_cast %arg11 : index to i32
              %28 = affine.apply #map3(%arg8)
              %29 = arith.cmpi eq, %28, %c0 : index
              %30 = arith.muli %27, %c4_i32 : i32
              %31 = arith.addi %30, %23 : i32
              %32 = arith.index_cast %31 : i32 to index
              %33 = memref.load %arg2[%32] : memref<?xf64>
              %34 = arith.muli %27, %c3_i32 : i32
              %35 = arith.addi %34, %23 : i32
              %36 = arith.index_cast %35 : i32 to index
              %37 = memref.load %arg0[%36] : memref<?xf64>
              %38 = arith.select %29, %33, %37 : f64
              %39 = polygeist.submap(%alloca_1, %c5) {map = #map2} : (memref<5xf64>, index) -> memref<?xf64>
              %40 = polygeist.submap(%alloca_2, %arg11, %c5) {map = #map4} : (memref<5x5xf64>, index, index) -> memref<?xf64>
              linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%39 : memref<?xf64>) outs(%40 : memref<?xf64>) {
              ^bb0(%in: f64, %out: f64):
                %41 = arith.mulf %in, %38 : f64
                %42 = arith.addf %out, %41 : f64
                linalg.yield %42 : f64
              }
            }
          }
          affine.for %arg10 = 0 to 5 {
            %23 = arith.index_cast %arg10 : index to i32
            %24 = arith.muli %23, %c4_i32 : i32
            %25 = arith.addi %24, %20 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = arith.muli %23, %c3_i32 : i32
            %28 = arith.addi %27, %20 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = affine.apply #map5(%arg8)
            %31 = arith.cmpi eq, %30, %c0 : index
            %32 = memref.load %arg2[%26] : memref<?xf64>
            %33 = memref.load %arg0[%29] : memref<?xf64>
            %34 = arith.select %31, %32, %33 : f64
            %35 = polygeist.submap(%alloca_2, %c5, %c5) {map = #map1} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
            %36 = polygeist.submap(%alloca_3, %arg10, %c5, %c5) {map = #map6} : (memref<5x5x5xf64>, index, index, index) -> memref<?x?xf64>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%35 : memref<?x?xf64>) outs(%36 : memref<?x?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %37 = arith.mulf %in, %34 : f64
              %38 = arith.addf %out, %37 : f64
              linalg.yield %38 : f64
            }
          }
        }
        %17 = arith.muli %13, %11 : i32
        %18 = arith.muli %17, %9 : i32
        %19 = arith.addi %6, %18 : i32
        affine.store %19, %alloca_4[] : memref<i32>
      }
      %3 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map7} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      %4 = polygeist.submap(%alloca_3, %c5, %c5, %c5) {map = #map} : (memref<5x5x5xf64>, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : memref<?x?x?xf64>) outs(%4 : memref<?x?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %6 = arith.mulf %out, %in : f64
        linalg.yield %6 : f64
      }
      %alloca_5 = memref.alloca(%c5) : memref<?xi32>
      %5 = polygeist.submap(%alloca_5, %c5) {map = #map2} : (memref<?xi32>, index) -> memref<?xi32>
      linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%5 : memref<?xi32>) {
      ^bb0(%out: i32):
        linalg.yield %c0_i32 : i32
      }
      affine.for %arg8 = 0 to 5 {
        %6 = arith.index_cast %arg8 : index to i32
        affine.for %arg9 = 0 to 3 {
          %7 = affine.load %alloca_5[%arg8] : memref<?xi32>
          %8 = arith.index_cast %arg9 : index to i32
          %9 = arith.cmpi eq, %8, %c2_i32 : i32
          %10 = arith.select %9, %c4_i32, %c3_i32 : i32
          %11 = arith.cmpi eq, %8, %c1_i32 : i32
          %12 = arith.select %11, %c4_i32, %c3_i32 : i32
          %13 = arith.cmpi eq, %8, %c0_i32 : i32
          %14 = arith.select %13, %c4_i32, %c3_i32 : i32
          %15 = arith.index_cast %12 : i32 to index
          %16 = arith.index_cast %14 : i32 to index
          scf.for %arg10 = %c0 to %15 step %c1 {
            scf.for %arg11 = %c0 to %16 step %c1 {
              memref.store %cst, %alloca_0[%arg10, %arg11] : memref<4x4xf64>
            }
          }
          %17 = arith.cmpi sgt, %16, %c0 : index
          affine.for %arg10 = 0 to 5 {
            %22 = arith.index_cast %arg10 : index to i32
            scf.for %arg11 = %c0 to %16 step %c1 {
              memref.store %cst, %alloca[%arg11] : memref<4xf64>
            }
            affine.for %arg11 = 0 to 5 {
              %23 = arith.index_cast %arg11 : index to i32
              %24 = affine.load %alloca_3[%arg8, %arg10, %arg11] : memref<5x5x5xf64>
              scf.for %arg12 = %c0 to %16 step %c1 {
                %25 = arith.index_cast %arg12 : index to i32
                %26 = arith.cmpi eq, %arg9, %c0 : index
                %27 = arith.muli %25, %c5_i32 : i32
                %28 = arith.addi %27, %23 : i32
                %29 = arith.index_cast %28 : i32 to index
                %30 = memref.load %arg3[%29] : memref<?xf64>
                %31 = arith.muli %25, %c5_i32 : i32
                %32 = arith.addi %31, %23 : i32
                %33 = arith.index_cast %32 : i32 to index
                %34 = memref.load %arg1[%33] : memref<?xf64>
                %35 = arith.select %26, %30, %34 : f64
                %36 = arith.mulf %24, %35 : f64
                %37 = memref.load %alloca[%arg12] : memref<4xf64>
                %38 = arith.addf %37, %36 : f64
                memref.store %38, %alloca[%arg12] : memref<4xf64>
              }
            }
            scf.for %arg11 = %c0 to %15 step %c1 {
              scf.if %17 {
                %23 = arith.index_cast %arg11 : index to i32
                %24 = affine.apply #map3(%arg9)
                %25 = arith.cmpi eq, %24, %c0 : index
                %26 = arith.muli %23, %c5_i32 : i32
                %27 = arith.addi %26, %22 : i32
                %28 = arith.index_cast %27 : i32 to index
                %29 = memref.load %arg3[%28] : memref<?xf64>
                %30 = arith.muli %23, %c5_i32 : i32
                %31 = arith.addi %30, %22 : i32
                %32 = arith.index_cast %31 : i32 to index
                %33 = memref.load %arg1[%32] : memref<?xf64>
                %34 = arith.select %25, %29, %33 : f64
                scf.for %arg12 = %c0 to %16 step %c1 {
                  %35 = memref.load %alloca[%arg12] : memref<4xf64>
                  %36 = arith.mulf %35, %34 : f64
                  %37 = memref.load %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                  %38 = arith.addf %37, %36 : f64
                  memref.store %38, %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                }
              }
            }
          }
          %18 = arith.index_cast %10 : i32 to index
          scf.for %arg10 = %c0 to %18 step %c1 {
            %22 = arith.index_cast %arg10 : index to i32
            %23 = arith.muli %22, %12 : i32
            %24 = arith.muli %22, %c5_i32 : i32
            %25 = arith.addi %24, %6 : i32
            %26 = arith.index_cast %25 : i32 to index
            scf.for %arg11 = %c0 to %15 step %c1 {
              %27 = arith.index_cast %arg11 : index to i32
              %28 = arith.addi %27, %23 : i32
              %29 = arith.muli %28, %14 : i32
              scf.for %arg12 = %c0 to %16 step %c1 {
                %30 = arith.index_cast %arg12 : index to i32
                %31 = arith.addi %30, %29 : i32
                %32 = arith.addi %31, %7 : i32
                %33 = arith.addi %32, %2 : i32
                %34 = arith.index_cast %33 : i32 to index
                %35 = memref.load %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                %36 = affine.apply #map5(%arg9)
                %37 = arith.cmpi eq, %36, %c0 : index
                %38 = memref.load %arg3[%26] : memref<?xf64>
                %39 = memref.load %arg1[%26] : memref<?xf64>
                %40 = arith.select %37, %38, %39 : f64
                %41 = arith.mulf %35, %40 : f64
                %42 = memref.load %arg6[%34] : memref<?xf64>
                %43 = arith.addf %42, %41 : f64
                memref.store %43, %arg6[%34] : memref<?xf64>
              }
            }
          }
          %19 = arith.muli %14, %12 : i32
          %20 = arith.muli %19, %10 : i32
          %21 = arith.addi %7, %20 : i32
          affine.store %21, %alloca_5[%arg8] : memref<?xi32>
        }
      }
    }
    return
  }
}
