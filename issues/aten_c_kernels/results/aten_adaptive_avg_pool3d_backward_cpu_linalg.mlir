#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c-11 = arith.constant -11 : index
    %c56 = arith.constant 56 : index
    %c336 = arith.constant 336 : index
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.muli %arg2, %c336 : index
      affine.for %arg3 = 0 to 3 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.muli %1, %c6_i32 : i32
        %3 = arith.divsi %2, %c3_i32 : i32
        %4 = arith.addi %1, %c1_i32 : i32
        %5 = arith.muli %4, %c6_i32 : i32
        %6 = arith.addi %5, %c2_i32 : i32
        %7 = arith.divsi %6, %c3_i32 : i32
        %8 = arith.subi %7, %3 : i32
        affine.for %arg4 = 0 to 3 {
          %9 = arith.index_cast %arg4 : index to i32
          %10 = arith.muli %9, %c7_i32 : i32
          %11 = arith.divsi %10, %c3_i32 : i32
          %12 = arith.addi %9, %c1_i32 : i32
          %13 = arith.muli %12, %c7_i32 : i32
          %14 = arith.addi %13, %c2_i32 : i32
          %15 = arith.divsi %14, %c3_i32 : i32
          %16 = arith.subi %15, %11 : i32
          %17 = arith.muli %8, %16 : i32
          %18 = arith.muli %arg4, %c7 : index
          %19 = arith.cmpi slt, %18, %c0 : index
          %20 = arith.subi %c-1, %18 : index
          %21 = arith.select %19, %20, %18 : index
          %22 = arith.divsi %21, %c3 : index
          %23 = arith.subi %c-1, %22 : index
          %24 = arith.select %19, %23, %22 : index
          %25 = arith.addi %24, %c3 : index
          affine.for %arg5 = 0 to 3 {
            %26 = arith.index_cast %arg5 : index to i32
            %27 = arith.muli %26, %c8_i32 : i32
            %28 = arith.divsi %27, %c3_i32 : i32
            %29 = arith.addi %26, %c1_i32 : i32
            %30 = arith.muli %29, %c8_i32 : i32
            %31 = arith.addi %30, %c2_i32 : i32
            %32 = arith.divsi %31, %c3_i32 : i32
            %33 = arith.subi %32, %28 : i32
            %34 = arith.muli %17, %33 : i32
            %35 = arith.sitofp %34 : i32 to f32
            %36 = arith.muli %arg5, %c8 : index
            %37 = arith.cmpi slt, %36, %c0 : index
            %38 = arith.subi %c-1, %36 : index
            %39 = arith.select %37, %38, %36 : index
            %40 = arith.divsi %39, %c3 : index
            %41 = arith.subi %c-1, %40 : index
            %42 = arith.select %37, %41, %40 : index
            %43 = arith.addi %36, %c10 : index
            %44 = arith.cmpi slt, %43, %c0 : index
            %45 = arith.subi %c-11, %36 : index
            %46 = arith.select %44, %45, %43 : index
            %47 = arith.divsi %46, %c3 : index
            %48 = arith.subi %c-1, %47 : index
            %49 = arith.select %44, %48, %47 : index
            affine.for %arg6 = #map1(%arg3) to #map2(%arg3) {
              %50 = arith.muli %arg6, %c56 : index
              scf.for %arg7 = %24 to %25 step %c1 {
                %51 = arith.muli %arg7, %c8 : index
                scf.for %arg8 = %42 to %49 step %c1 {
                  %52 = affine.load %arg0[%arg2 * 27 + %arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xf32>
                  %53 = arith.divf %52, %35 : f32
                  %54 = arith.addi %arg8, %51 : index
                  %55 = arith.addi %54, %0 : index
                  %56 = arith.addi %55, %50 : index
                  %57 = memref.load %arg1[%56] : memref<?xf32>
                  %58 = arith.addf %57, %53 : f32
                  memref.store %58, %arg1[%56] : memref<?xf32>
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

