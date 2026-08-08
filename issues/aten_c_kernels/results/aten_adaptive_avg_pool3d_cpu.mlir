#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-11 = arith.constant -11 : index
    %c56 = arith.constant 56 : index
    %c336 = arith.constant 336 : index
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg2 = 0 to 2 {
      %0 = arith.muli %arg2, %c336 : index
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
          %1 = arith.muli %arg4, %c7 : index
          %2 = arith.cmpi slt, %1, %c0 : index
          %3 = arith.subi %c-1, %1 : index
          %4 = arith.select %2, %3, %1 : index
          %5 = arith.divsi %4, %c3 : index
          %6 = arith.subi %c-1, %5 : index
          %7 = arith.select %2, %6, %5 : index
          %8 = arith.addi %7, %c3 : index
          affine.for %arg5 = 0 to 3 {
            %9 = arith.index_cast %arg5 : index to i32
            %10 = arith.muli %9, %c8_i32 : i32
            %11 = arith.divsi %10, %c3_i32 : i32
            %12 = arith.addi %9, %c1_i32 : i32
            %13 = arith.muli %12, %c8_i32 : i32
            %14 = arith.addi %13, %c2_i32 : i32
            %15 = arith.divsi %14, %c3_i32 : i32
            %16 = arith.index_cast %15 : i32 to index
            %17 = arith.index_cast %11 : i32 to index
            %18 = arith.subi %16, %17 : index
            %19 = arith.muli %arg5, %c8 : index
            %20 = arith.cmpi slt, %19, %c0 : index
            %21 = arith.subi %c-1, %19 : index
            %22 = arith.select %20, %21, %19 : index
            %23 = arith.divsi %22, %c3 : index
            %24 = arith.subi %c-1, %23 : index
            %25 = arith.select %20, %24, %23 : index
            %26 = arith.addi %19, %c10 : index
            %27 = arith.cmpi slt, %26, %c0 : index
            %28 = arith.subi %c-11, %19 : index
            %29 = arith.select %27, %28, %26 : index
            %30 = arith.divsi %29, %c3 : index
            %31 = arith.subi %c-1, %30 : index
            %32 = arith.select %27, %31, %30 : index
            %33:2 = affine.for %arg6 = #map(%arg3) to #map1(%arg3) iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f32) {
              %36 = arith.muli %arg6, %c56 : index
              %37:2 = scf.for %arg9 = %7 to %8 step %c1 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (i32, f32) {
                %38 = arith.index_cast %arg10 : i32 to index
                %39 = arith.addi %38, %18 : index
                %40 = arith.index_cast %39 : index to i32
                %41 = arith.muli %arg9, %c8 : index
                %42 = scf.for %arg12 = %25 to %32 step %c1 iter_args(%arg13 = %arg11) -> (f32) {
                  %43 = arith.addi %arg12, %41 : index
                  %44 = arith.addi %43, %0 : index
                  %45 = arith.addi %44, %36 : index
                  %46 = memref.load %arg0[%45] : memref<?xf32>
                  %47 = arith.addf %arg13, %46 : f32
                  scf.yield %47 : f32
                }
                scf.yield %40, %42 : i32, f32
              }
              affine.yield %37#0, %37#1 : i32, f32
            }
            %34 = arith.sitofp %33#0 : i32 to f32
            %35 = arith.divf %33#1, %34 : f32
            affine.store %35, %arg1[%arg2 * 27 + %arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}
