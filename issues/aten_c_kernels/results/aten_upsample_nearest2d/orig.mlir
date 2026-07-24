module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest2d(%arg0: memref<?x4x8x8xf32>, %arg1: memref<?x4x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 16 {
          %0 = arith.cmpi slt, %arg4, %c0 : index
          %1 = arith.subi %c-1, %arg4 : index
          %2 = arith.select %0, %1, %arg4 : index
          %3 = arith.divsi %2, %c2 : index
          %4 = arith.subi %c-1, %3 : index
          %5 = arith.select %0, %4, %3 : index
          affine.for %arg5 = 0 to 16 {
            %6 = arith.cmpi slt, %arg5, %c0 : index
            %7 = arith.subi %c-1, %arg5 : index
            %8 = arith.select %6, %7, %arg5 : index
            %9 = arith.divsi %8, %c2 : index
            %10 = arith.subi %c-1, %9 : index
            %11 = arith.select %6, %10, %9 : index
            %12 = memref.load %arg0[%arg2, %arg3, %5, %11] : memref<?x4x8x8xf32>
            affine.store %12, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x4x16x16xf32>
          }
        }
      }
    }
    return
  }
}
