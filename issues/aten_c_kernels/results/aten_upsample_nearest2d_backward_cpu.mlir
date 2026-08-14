module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    affine.for %arg2 = 0 to 40 {
      affine.store %cst, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.muli %2, %c4_i32 : i32
        %4 = arith.divsi %3, %c7_i32 : i32
        %5 = arith.cmpi sge, %4, %c4_i32 : i32
        %6 = arith.select %5, %c3_i32, %4 : i32
        %7 = arith.addi %1, %6 : i32
        %8 = arith.muli %7, %c5_i32 : i32
        affine.for %arg4 = 0 to 8 {
          %9 = arith.index_cast %arg4 : index to i32
          %10 = arith.muli %9, %c5_i32 : i32
          %11 = arith.divsi %10, %c8_i32 : i32
          %12 = arith.cmpi sge, %11, %c5_i32 : i32
          %13 = arith.select %12, %c4_i32, %11 : i32
          %14 = arith.addi %8, %13 : i32
          %15 = arith.index_cast %14 : i32 to index
          %16 = affine.load %arg0[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
          %17 = memref.load %arg1[%15] : memref<?xf32>
          %18 = arith.addf %17, %16 : f32
          memref.store %18, %arg1[%15] : memref<?xf32>
        }
      }
    }
    return
  }
}
