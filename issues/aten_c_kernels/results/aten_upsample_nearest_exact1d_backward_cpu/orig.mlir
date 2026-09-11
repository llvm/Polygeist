module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest_exact1d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %c14_i32 = arith.constant 14 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 8 {
      affine.store %cst, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.muli %2, %c2_i32 : i32
        %4 = arith.addi %3, %c1_i32 : i32
        %5 = arith.muli %4, %c4_i32 : i32
        %6 = arith.divsi %5, %c14_i32 : i32
        %7 = arith.cmpi sge, %6, %c4_i32 : i32
        %8 = arith.select %7, %c3_i32, %6 : i32
        %9 = arith.addi %1, %8 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = affine.load %arg0[%arg3 + %arg2 * 7] : memref<?xf32>
        %12 = memref.load %arg1[%10] : memref<?xf32>
        %13 = arith.addf %12, %11 : f32
        memref.store %13, %arg1[%10] : memref<?xf32>
      }
    }
    return
  }
}
