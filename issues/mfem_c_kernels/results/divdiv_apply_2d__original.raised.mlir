#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (s0, d0)>
#map3 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 25 + d0 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_divdiv_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c24_i32 = arith.constant 24 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<5xf64>
    %alloca_1 = memref.alloca() : memref<5x5xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = arith.index_cast %arg7 : index to i32
      %1 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%1 : memref<?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %2 = arith.muli %0, %c24_i32 : i32
      %alloca_2 = memref.alloca() : memref<i32>
      affine.store %c0_i32, %alloca_2[] : memref<i32>
      affine.for %arg8 = 0 to 2 {
        %6 = affine.load %alloca_2[] : memref<i32>
        %7 = arith.index_cast %arg8 : index to i32
        %8 = arith.cmpi eq, %7, %c1_i32 : i32
        %9 = arith.select %8, %c3_i32, %c4_i32 : i32
        %10 = arith.cmpi eq, %7, %c0_i32 : i32
        %11 = arith.select %10, %c3_i32, %c4_i32 : i32
        %12 = arith.index_cast %11 : i32 to index
        %13 = arith.index_cast %9 : i32 to index
        scf.for %arg9 = %c0 to %12 step %c1 {
          %16 = arith.index_cast %arg9 : index to i32
          %17 = polygeist.submap(%alloca_0, %c5) {map = #map1} : (memref<5xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%17 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %18 = arith.muli %16, %9 : i32
          scf.for %arg10 = %c0 to %13 step %c1 {
            %19 = arith.index_cast %arg10 : index to i32
            %20 = arith.addi %19, %18 : i32
            %21 = arith.addi %20, %6 : i32
            %22 = arith.addi %21, %2 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = memref.load %arg5[%23] : memref<?xf64>
            %25 = polygeist.submap(%alloca_0, %c5) {map = #map1} : (memref<5xf64>, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%25 : memref<?xf64>) {
            ^bb0(%out: f64):
              %26 = linalg.index 0 : index
              %27 = arith.index_cast %26 : index to i32
              %28 = arith.cmpi eq, %arg8, %c0 : index
              %29 = arith.muli %27, %c4_i32 : i32
              %30 = arith.addi %29, %19 : i32
              %31 = arith.index_cast %30 : i32 to index
              %32 = memref.load %arg2[%31] : memref<?xf64>
              %33 = arith.muli %27, %c3_i32 : i32
              %34 = arith.addi %33, %19 : i32
              %35 = arith.index_cast %34 : i32 to index
              %36 = memref.load %arg0[%35] : memref<?xf64>
              %37 = arith.select %28, %32, %36 : f64
              %38 = arith.mulf %24, %37 : f64
              %39 = arith.addf %out, %38 : f64
              linalg.yield %39 : f64
            }
          }
          affine.for %arg10 = 0 to 5 {
            %19 = arith.index_cast %arg10 : index to i32
            %20 = arith.cmpi eq, %arg8, %c0 : index
            %21 = arith.muli %19, %c3_i32 : i32
            %22 = arith.addi %21, %16 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = memref.load %arg0[%23] : memref<?xf64>
            %25 = arith.muli %19, %c4_i32 : i32
            %26 = arith.addi %25, %16 : i32
            %27 = arith.index_cast %26 : i32 to index
            %28 = memref.load %arg2[%27] : memref<?xf64>
            %29 = arith.select %20, %24, %28 : f64
            %30 = polygeist.submap(%alloca_0, %c5) {map = #map1} : (memref<5xf64>, index) -> memref<?xf64>
            %31 = polygeist.submap(%alloca_1, %arg10, %c5) {map = #map2} : (memref<5x5xf64>, index, index) -> memref<?xf64>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%30 : memref<?xf64>) outs(%31 : memref<?xf64>) {
            ^bb0(%in: f64, %out: f64):
              %32 = arith.mulf %in, %29 : f64
              %33 = arith.addf %out, %32 : f64
              linalg.yield %33 : f64
            }
          }
        }
        %14 = arith.muli %9, %11 : i32
        %15 = arith.addi %6, %14 : i32
        affine.store %15, %alloca_2[] : memref<i32>
      }
      %3 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map3} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %4 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : memref<?x?xf64>) outs(%4 : memref<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %6 = arith.mulf %out, %in : f64
        linalg.yield %6 : f64
      }
      %alloca_3 = memref.alloca(%c5) : memref<?xi32>
      %5 = polygeist.submap(%alloca_3, %c5) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%5 : memref<?xi32>) {
      ^bb0(%out: i32):
        linalg.yield %c0_i32 : i32
      }
      affine.for %arg8 = 0 to 5 {
        %6 = arith.index_cast %arg8 : index to i32
        affine.for %arg9 = 0 to 2 {
          %7 = affine.load %alloca_3[%arg8] : memref<?xi32>
          %8 = arith.index_cast %arg9 : index to i32
          %9 = arith.cmpi eq, %8, %c1_i32 : i32
          %10 = arith.select %9, %c3_i32, %c4_i32 : i32
          %11 = arith.cmpi eq, %8, %c0_i32 : i32
          %12 = arith.select %11, %c3_i32, %c4_i32 : i32
          %13 = arith.index_cast %10 : i32 to index
          scf.for %arg10 = %c0 to %13 step %c1 {
            memref.store %cst, %alloca[%arg10] : memref<4xf64>
          }
          affine.for %arg10 = 0 to 5 {
            %17 = arith.index_cast %arg10 : index to i32
            %18 = affine.load %alloca_1[%arg8, %arg10] : memref<5x5xf64>
            scf.for %arg11 = %c0 to %13 step %c1 {
              %19 = arith.index_cast %arg11 : index to i32
              %20 = arith.cmpi eq, %arg9, %c0 : index
              %21 = arith.muli %19, %c5_i32 : i32
              %22 = arith.addi %21, %17 : i32
              %23 = arith.index_cast %22 : i32 to index
              %24 = memref.load %arg3[%23] : memref<?xf64>
              %25 = arith.muli %19, %c5_i32 : i32
              %26 = arith.addi %25, %17 : i32
              %27 = arith.index_cast %26 : i32 to index
              %28 = memref.load %arg1[%27] : memref<?xf64>
              %29 = arith.select %20, %24, %28 : f64
              %30 = arith.mulf %18, %29 : f64
              %31 = memref.load %alloca[%arg11] : memref<4xf64>
              %32 = arith.addf %31, %30 : f64
              memref.store %32, %alloca[%arg11] : memref<4xf64>
            }
          }
          %14 = arith.index_cast %12 : i32 to index
          scf.for %arg10 = %c0 to %14 step %c1 {
            %17 = arith.index_cast %arg10 : index to i32
            %18 = arith.cmpi eq, %arg9, %c0 : index
            %19 = arith.muli %17, %c5_i32 : i32
            %20 = arith.addi %19, %6 : i32
            %21 = arith.index_cast %20 : i32 to index
            %22 = memref.load %arg1[%21] : memref<?xf64>
            %23 = arith.muli %17, %c5_i32 : i32
            %24 = arith.addi %23, %6 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = memref.load %arg3[%25] : memref<?xf64>
            %27 = arith.select %18, %22, %26 : f64
            %28 = arith.muli %17, %10 : i32
            scf.for %arg11 = %c0 to %13 step %c1 {
              %29 = arith.index_cast %arg11 : index to i32
              %30 = arith.addi %29, %28 : i32
              %31 = arith.addi %30, %7 : i32
              %32 = arith.addi %31, %2 : i32
              %33 = arith.index_cast %32 : i32 to index
              %34 = memref.load %alloca[%arg11] : memref<4xf64>
              %35 = arith.mulf %34, %27 : f64
              %36 = memref.load %arg6[%33] : memref<?xf64>
              %37 = arith.addf %36, %35 : f64
              memref.store %37, %arg6[%33] : memref<?xf64>
            }
          }
          %15 = arith.muli %10, %12 : i32
          %16 = arith.addi %7, %15 : i32
          affine.store %16, %alloca_3[%arg8] : memref<?xi32>
        }
      }
    }
    return
  }
}
