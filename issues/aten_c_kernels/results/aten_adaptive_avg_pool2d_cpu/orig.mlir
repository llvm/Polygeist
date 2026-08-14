#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c42 = arith.constant 42 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg2 = 0 to 2 {
      %0 = arith.muli %arg2, %c42 : index
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
          %1 = arith.index_cast %arg4 : index to i32
          %2 = arith.muli %1, %c7_i32 : i32
          %3 = arith.divsi %2, %c3_i32 : i32
          %4 = arith.addi %1, %c1_i32 : i32
          %5 = arith.muli %4, %c7_i32 : i32
          %6 = arith.addi %5, %c2_i32 : i32
          %7 = arith.divsi %6, %c3_i32 : i32
          %8 = arith.index_cast %7 : i32 to index
          %9 = arith.index_cast %3 : i32 to index
          %10 = arith.subi %8, %9 : index
          %11 = arith.muli %arg4, %c7 : index
          %12 = arith.cmpi slt, %11, %c0 : index
          %13 = arith.subi %c-1, %11 : index
          %14 = arith.select %12, %13, %11 : index
          %15 = arith.divsi %14, %c3 : index
          %16 = arith.subi %c-1, %15 : index
          %17 = arith.select %12, %16, %15 : index
          %18 = arith.addi %17, %c3 : index
          %19:2 = affine.for %arg5 = #map(%arg3) to #map1(%arg3) iter_args(%arg6 = %c0_i32, %arg7 = %cst) -> (i32, f32) {
            %22 = arith.index_cast %arg6 : i32 to index
            %23 = arith.addi %22, %10 : index
            %24 = arith.index_cast %23 : index to i32
            %25 = arith.muli %arg5, %c7 : index
            %26 = scf.for %arg8 = %17 to %18 step %c1 iter_args(%arg9 = %arg7) -> (f32) {
              %27 = arith.addi %arg8, %0 : index
              %28 = arith.addi %27, %25 : index
              %29 = memref.load %arg0[%28] : memref<?xf32>
              %30 = arith.addf %arg9, %29 : f32
              scf.yield %30 : f32
            }
            affine.yield %24, %26 : i32, f32
          }
          %20 = arith.sitofp %19#0 : i32 to f32
          %21 = arith.divf %19#1, %20 : f32
          affine.store %21, %arg1[%arg4 + %arg2 * 9 + %arg3 * 3] : memref<?xf32>
        }
      }
    }
    return
  }
}
