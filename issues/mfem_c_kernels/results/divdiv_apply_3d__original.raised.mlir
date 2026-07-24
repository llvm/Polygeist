#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 - 1)>
#map4 = affine_map<(d0) -> (d0 - 2)>
#map5 = affine_map<(d0, d1, d2)[s0] -> (d2 + s0 * 125 + d0 * 25 + d1 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloca_3 : memref<5x5x5xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %1 = arith.muli %0, %c108_i32 : i32
      %alloca_4 = memref.alloca() : memref<i32>
      affine.store %c0_i32, %alloca_4[] : memref<i32>
      affine.for %arg8 = 0 to 3 {
        %3 = affine.load %alloca_4[] : memref<i32>
        %4 = arith.index_cast %arg8 : index to i32
        %5 = arith.cmpi eq, %4, %c2_i32 : i32
        %6 = arith.select %5, %c4_i32, %c3_i32 : i32
        %7 = arith.cmpi eq, %4, %c1_i32 : i32
        %8 = arith.select %7, %c4_i32, %c3_i32 : i32
        %9 = arith.cmpi eq, %4, %c0_i32 : i32
        %10 = arith.select %9, %c4_i32, %c3_i32 : i32
        %11 = arith.index_cast %6 : i32 to index
        %12 = arith.index_cast %8 : i32 to index
        %13 = arith.index_cast %10 : i32 to index
        scf.for %arg9 = %c0 to %11 step %c1 {
          %17 = arith.index_cast %arg9 : index to i32
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%alloca_2 : memref<5x5xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst : f64
          }
          %18 = arith.muli %17, %8 : i32
          scf.for %arg10 = %c0 to %12 step %c1 {
            %19 = arith.index_cast %arg10 : index to i32
            linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_1 : memref<5xf64>) {
            ^bb0(%out: f64):
              linalg.yield %cst : f64
            }
            %20 = arith.addi %19, %18 : i32
            %21 = arith.muli %20, %10 : i32
            scf.for %arg11 = %c0 to %13 step %c1 {
              %22 = arith.index_cast %arg11 : index to i32
              %23 = arith.addi %22, %21 : i32
              %24 = arith.addi %23, %3 : i32
              %25 = arith.addi %24, %1 : i32
              %26 = arith.index_cast %25 : i32 to index
              %27 = memref.load %arg5[%26] : memref<?xf64>
              linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%alloca_1 : memref<5xf64>) {
              ^bb0(%out: f64):
                %28 = linalg.index 0 : index
                %29 = arith.index_cast %28 : index to i32
                %30 = arith.cmpi eq, %arg8, %c0 : index
                %31 = arith.muli %29, %c4_i32 : i32
                %32 = arith.addi %31, %22 : i32
                %33 = arith.index_cast %32 : i32 to index
                %34 = memref.load %arg2[%33] : memref<?xf64>
                %35 = arith.muli %29, %c3_i32 : i32
                %36 = arith.addi %35, %22 : i32
                %37 = arith.index_cast %36 : i32 to index
                %38 = memref.load %arg0[%37] : memref<?xf64>
                %39 = arith.select %30, %34, %38 : f64
                %40 = arith.mulf %27, %39 : f64
                %41 = arith.addf %out, %40 : f64
                linalg.yield %41 : f64
              }
            }
            affine.for %arg11 = 0 to 5 {
              %22 = arith.index_cast %arg11 : index to i32
              %23 = affine.apply #map3(%arg8)
              %24 = arith.cmpi eq, %23, %c0 : index
              %25 = arith.muli %22, %c4_i32 : i32
              %26 = arith.addi %25, %19 : i32
              %27 = arith.index_cast %26 : i32 to index
              %28 = memref.load %arg2[%27] : memref<?xf64>
              %29 = arith.muli %22, %c3_i32 : i32
              %30 = arith.addi %29, %19 : i32
              %31 = arith.index_cast %30 : i32 to index
              %32 = memref.load %arg0[%31] : memref<?xf64>
              %33 = arith.select %24, %28, %32 : f64
              %subview = memref.subview %alloca_1[0] [%c5] [1] : memref<5xf64> to memref<?xf64, strided<[1]>>
              %subview_5 = memref.subview %alloca_2[%arg11, 0] [1, %c5] [1, 1] : memref<5x5xf64> to memref<?xf64, strided<[1], offset: ?>>
              linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%subview : memref<?xf64, strided<[1]>>) outs(%subview_5 : memref<?xf64, strided<[1], offset: ?>>) {
              ^bb0(%in: f64, %out: f64):
                %34 = arith.mulf %in, %33 : f64
                %35 = arith.addf %out, %34 : f64
                linalg.yield %35 : f64
              }
            }
          }
          affine.for %arg10 = 0 to 5 {
            %19 = arith.index_cast %arg10 : index to i32
            %20 = arith.muli %19, %c4_i32 : i32
            %21 = arith.addi %20, %17 : i32
            %22 = arith.index_cast %21 : i32 to index
            %23 = arith.muli %19, %c3_i32 : i32
            %24 = arith.addi %23, %17 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.apply #map4(%arg8)
            %27 = arith.cmpi eq, %26, %c0 : index
            %28 = memref.load %arg2[%22] : memref<?xf64>
            %29 = memref.load %arg0[%25] : memref<?xf64>
            %30 = arith.select %27, %28, %29 : f64
            %subview = memref.subview %alloca_2[0, 0] [%c5, %c5] [1, 1] : memref<5x5xf64> to memref<?x?xf64, strided<[5, 1]>>
            %subview_5 = memref.subview %alloca_3[%arg10, 0, 0] [1, %c5, %c5] [1, 1, 1] : memref<5x5x5xf64> to memref<?x?xf64, strided<[5, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<?x?xf64, strided<[5, 1]>>) outs(%subview_5 : memref<?x?xf64, strided<[5, 1], offset: ?>>) {
            ^bb0(%in: f64, %out: f64):
              %31 = arith.mulf %in, %30 : f64
              %32 = arith.addf %out, %31 : f64
              linalg.yield %32 : f64
            }
          }
        }
        %14 = arith.muli %10, %8 : i32
        %15 = arith.muli %14, %6 : i32
        %16 = arith.addi %3, %15 : i32
        affine.store %16, %alloca_4[] : memref<i32>
      }
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5, %c5) {map = #map5} : (memref<?xf64>, index, index, index, index) -> memref<?x?x?xf64>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : memref<?x?x?xf64>) outs(%alloca_3 : memref<5x5x5xf64>) {
      ^bb0(%in: f64, %out: f64):
        %3 = arith.mulf %out, %in : f64
        linalg.yield %3 : f64
      }
      affine.for %arg8 = 0 to 5 {
        %3 = arith.index_cast %arg8 : index to i32
        %alloca_5 = memref.alloca() : memref<i32>
        affine.store %c0_i32, %alloca_5[] : memref<i32>
        affine.for %arg9 = 0 to 3 {
          %4 = affine.load %alloca_5[] : memref<i32>
          %5 = arith.index_cast %arg9 : index to i32
          %6 = arith.cmpi eq, %5, %c2_i32 : i32
          %7 = arith.select %6, %c4_i32, %c3_i32 : i32
          %8 = arith.cmpi eq, %5, %c1_i32 : i32
          %9 = arith.select %8, %c4_i32, %c3_i32 : i32
          %10 = arith.cmpi eq, %5, %c0_i32 : i32
          %11 = arith.select %10, %c4_i32, %c3_i32 : i32
          %12 = arith.index_cast %9 : i32 to index
          %13 = arith.index_cast %11 : i32 to index
          scf.for %arg10 = %c0 to %12 step %c1 {
            scf.for %arg11 = %c0 to %13 step %c1 {
              memref.store %cst, %alloca_0[%arg10, %arg11] : memref<4x4xf64>
            }
          }
          %14 = arith.cmpi sgt, %13, %c0 : index
          affine.for %arg10 = 0 to 5 {
            %19 = arith.index_cast %arg10 : index to i32
            scf.for %arg11 = %c0 to %13 step %c1 {
              memref.store %cst, %alloca[%arg11] : memref<4xf64>
            }
            affine.for %arg11 = 0 to 5 {
              %20 = arith.index_cast %arg11 : index to i32
              %21 = affine.load %alloca_3[%arg8, %arg10, %arg11] : memref<5x5x5xf64>
              scf.for %arg12 = %c0 to %13 step %c1 {
                %22 = arith.index_cast %arg12 : index to i32
                %23 = arith.cmpi eq, %arg9, %c0 : index
                %24 = arith.muli %22, %c5_i32 : i32
                %25 = arith.addi %24, %20 : i32
                %26 = arith.index_cast %25 : i32 to index
                %27 = memref.load %arg3[%26] : memref<?xf64>
                %28 = arith.muli %22, %c5_i32 : i32
                %29 = arith.addi %28, %20 : i32
                %30 = arith.index_cast %29 : i32 to index
                %31 = memref.load %arg1[%30] : memref<?xf64>
                %32 = arith.select %23, %27, %31 : f64
                %33 = arith.mulf %21, %32 : f64
                %34 = memref.load %alloca[%arg12] : memref<4xf64>
                %35 = arith.addf %34, %33 : f64
                memref.store %35, %alloca[%arg12] : memref<4xf64>
              }
            }
            scf.for %arg11 = %c0 to %12 step %c1 {
              scf.if %14 {
                %20 = arith.index_cast %arg11 : index to i32
                %21 = affine.apply #map3(%arg9)
                %22 = arith.cmpi eq, %21, %c0 : index
                %23 = arith.muli %20, %c5_i32 : i32
                %24 = arith.addi %23, %19 : i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %arg3[%25] : memref<?xf64>
                %27 = arith.muli %20, %c5_i32 : i32
                %28 = arith.addi %27, %19 : i32
                %29 = arith.index_cast %28 : i32 to index
                %30 = memref.load %arg1[%29] : memref<?xf64>
                %31 = arith.select %22, %26, %30 : f64
                scf.for %arg12 = %c0 to %13 step %c1 {
                  %32 = memref.load %alloca[%arg12] : memref<4xf64>
                  %33 = arith.mulf %32, %31 : f64
                  %34 = memref.load %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                  %35 = arith.addf %34, %33 : f64
                  memref.store %35, %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                }
              }
            }
          }
          %15 = arith.index_cast %7 : i32 to index
          scf.for %arg10 = %c0 to %15 step %c1 {
            %19 = arith.index_cast %arg10 : index to i32
            %20 = arith.muli %19, %9 : i32
            %21 = arith.muli %19, %c5_i32 : i32
            %22 = arith.addi %21, %3 : i32
            %23 = arith.index_cast %22 : i32 to index
            scf.for %arg11 = %c0 to %12 step %c1 {
              %24 = arith.index_cast %arg11 : index to i32
              %25 = arith.addi %24, %20 : i32
              %26 = arith.muli %25, %11 : i32
              scf.for %arg12 = %c0 to %13 step %c1 {
                %27 = arith.index_cast %arg12 : index to i32
                %28 = arith.addi %27, %26 : i32
                %29 = arith.addi %28, %4 : i32
                %30 = arith.addi %29, %1 : i32
                %31 = arith.index_cast %30 : i32 to index
                %32 = memref.load %alloca_0[%arg11, %arg12] : memref<4x4xf64>
                %33 = affine.apply #map4(%arg9)
                %34 = arith.cmpi eq, %33, %c0 : index
                %35 = memref.load %arg3[%23] : memref<?xf64>
                %36 = memref.load %arg1[%23] : memref<?xf64>
                %37 = arith.select %34, %35, %36 : f64
                %38 = arith.mulf %32, %37 : f64
                %39 = memref.load %arg6[%31] : memref<?xf64>
                %40 = arith.addf %39, %38 : f64
                memref.store %40, %arg6[%31] : memref<?xf64>
              }
            }
          }
          %16 = arith.muli %11, %9 : i32
          %17 = arith.muli %16, %7 : i32
          %18 = arith.addi %4, %17 : i32
          affine.store %18, %alloca_5[] : memref<i32>
        }
      }
    }
    return
  }
}
