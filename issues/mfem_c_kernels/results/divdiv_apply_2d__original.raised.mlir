#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 25 + d0 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloca_1 : memref<5x5xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %1 = arith.muli %0, %c24_i32 : i32
      %alloca_2 = memref.alloca() : memref<i32>
      affine.store %c0_i32, %alloca_2[] : memref<i32>
      affine.for %arg8 = 0 to 2 {
        %3 = affine.load %alloca_2[] : memref<i32>
        %4 = arith.index_cast %arg8 : index to i32
        %5 = arith.cmpi eq, %4, %c1_i32 : i32
        %6 = arith.select %5, %c3_i32, %c4_i32 : i32
        %7 = arith.cmpi eq, %4, %c0_i32 : i32
        %8 = arith.select %7, %c3_i32, %c4_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.index_cast %6 : i32 to index
        scf.for %arg9 = %c0 to %9 step %c1 {
          %13 = arith.index_cast %arg9 : index to i32
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloca_0 : memref<5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %14 = arith.muli %13, %6 : i32
          scf.for %arg10 = %c0 to %10 step %c1 {
            %15 = arith.index_cast %arg10 : index to i32
            %16 = arith.addi %15, %14 : i32
            %17 = arith.addi %16, %3 : i32
            %18 = arith.addi %17, %1 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = memref.load %arg5[%19] : memref<?xf64>
            linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloca_0 : memref<5xf64>) {
            ^bb0(%out: f64):
              %21 = linalg.index 0 : index
              %22 = arith.index_cast %21 : index to i32
              %23 = arith.cmpi eq, %arg8, %c0 : index
              %24 = arith.muli %22, %c4_i32 : i32
              %25 = arith.addi %24, %15 : i32
              %26 = arith.index_cast %25 : i32 to index
              %27 = memref.load %arg2[%26] : memref<?xf64>
              %28 = arith.muli %22, %c3_i32 : i32
              %29 = arith.addi %28, %15 : i32
              %30 = arith.index_cast %29 : i32 to index
              %31 = memref.load %arg0[%30] : memref<?xf64>
              %32 = arith.select %23, %27, %31 : f64
              %33 = arith.mulf %20, %32 : f64
              %34 = arith.addf %out, %33 : f64
              linalg.yield %34 : f64
            }
          }
          affine.for %arg10 = 0 to 5 {
            %15 = arith.index_cast %arg10 : index to i32
            %16 = arith.cmpi eq, %arg8, %c0 : index
            %17 = arith.muli %15, %c3_i32 : i32
            %18 = arith.addi %17, %13 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = memref.load %arg0[%19] : memref<?xf64>
            %21 = arith.muli %15, %c4_i32 : i32
            %22 = arith.addi %21, %13 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = memref.load %arg2[%23] : memref<?xf64>
            %25 = arith.select %16, %20, %24 : f64
            %subview = memref.subview %alloca_0[0] [%c5] [1] : memref<5xf64> to memref<?xf64, strided<[1]>>
            %subview_3 = memref.subview %alloca_1[%arg10, 0] [1, %c5] [1, 1] : memref<5x5xf64> to memref<?xf64, strided<[1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%subview : memref<?xf64, strided<[1]>>) outs(%subview_3 : memref<?xf64, strided<[1], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %26 = arith.mulf %in, %25 : f64
              %27 = arith.addf %out, %26 : f64
              linalg.yield %27 : f64
            }
          }
        }
        %11 = arith.muli %6, %8 : i32
        %12 = arith.addi %3, %11 : i32
        affine.store %12, %alloca_2[] : memref<i32>
      }
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map2} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<?x?xf64>) outs(%alloca_1 : memref<5x5xf64>) {
      ^bb0(%in: f64, %out: f64):
        %3 = arith.mulf %out, %in : f64
        linalg.yield %3 : f64
      }
      affine.for %arg8 = 0 to 5 {
        %3 = arith.index_cast %arg8 : index to i32
        %alloca_3 = memref.alloca() : memref<i32>
        affine.store %c0_i32, %alloca_3[] : memref<i32>
        affine.for %arg9 = 0 to 2 {
          %4 = affine.load %alloca_3[] : memref<i32>
          %5 = arith.index_cast %arg9 : index to i32
          %6 = arith.cmpi eq, %5, %c1_i32 : i32
          %7 = arith.select %6, %c3_i32, %c4_i32 : i32
          %8 = arith.cmpi eq, %5, %c0_i32 : i32
          %9 = arith.select %8, %c3_i32, %c4_i32 : i32
          %10 = arith.index_cast %7 : i32 to index
          scf.for %arg10 = %c0 to %10 step %c1 {
            memref.store %cst, %alloca[%arg10] : memref<4xf64>
          }
          affine.for %arg10 = 0 to 5 {
            %14 = arith.index_cast %arg10 : index to i32
            %15 = affine.load %alloca_1[%arg8, %arg10] : memref<5x5xf64>
            scf.for %arg11 = %c0 to %10 step %c1 {
              %16 = arith.index_cast %arg11 : index to i32
              %17 = arith.cmpi eq, %arg9, %c0 : index
              %18 = arith.muli %16, %c5_i32 : i32
              %19 = arith.addi %18, %14 : i32
              %20 = arith.index_cast %19 : i32 to index
              %21 = memref.load %arg3[%20] : memref<?xf64>
              %22 = arith.muli %16, %c5_i32 : i32
              %23 = arith.addi %22, %14 : i32
              %24 = arith.index_cast %23 : i32 to index
              %25 = memref.load %arg1[%24] : memref<?xf64>
              %26 = arith.select %17, %21, %25 : f64
              %27 = arith.mulf %15, %26 : f64
              %28 = memref.load %alloca[%arg11] : memref<4xf64>
              %29 = arith.addf %28, %27 : f64
              memref.store %29, %alloca[%arg11] : memref<4xf64>
            }
          }
          %11 = arith.index_cast %9 : i32 to index
          scf.for %arg10 = %c0 to %11 step %c1 {
            %14 = arith.index_cast %arg10 : index to i32
            %15 = arith.cmpi eq, %arg9, %c0 : index
            %16 = arith.muli %14, %c5_i32 : i32
            %17 = arith.addi %16, %3 : i32
            %18 = arith.index_cast %17 : i32 to index
            %19 = memref.load %arg1[%18] : memref<?xf64>
            %20 = arith.muli %14, %c5_i32 : i32
            %21 = arith.addi %20, %3 : i32
            %22 = arith.index_cast %21 : i32 to index
            %23 = memref.load %arg3[%22] : memref<?xf64>
            %24 = arith.select %15, %19, %23 : f64
            %25 = arith.muli %14, %7 : i32
            scf.for %arg11 = %c0 to %10 step %c1 {
              %26 = arith.index_cast %arg11 : index to i32
              %27 = arith.addi %26, %25 : i32
              %28 = arith.addi %27, %4 : i32
              %29 = arith.addi %28, %1 : i32
              %30 = arith.index_cast %29 : i32 to index
              %31 = memref.load %alloca[%arg11] : memref<4xf64>
              %32 = arith.mulf %31, %24 : f64
              %33 = memref.load %arg6[%30] : memref<?xf64>
              %34 = arith.addf %33, %32 : f64
              memref.store %34, %arg6[%30] : memref<?xf64>
            }
          }
          %12 = arith.muli %7, %9 : i32
          %13 = arith.addi %4, %12 : i32
          affine.store %13, %alloca_3[] : memref<i32>
        }
      }
    }
    return
  }
}
