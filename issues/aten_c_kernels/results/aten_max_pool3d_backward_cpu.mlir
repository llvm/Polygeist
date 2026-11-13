module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c336_i32 = arith.constant 336 : i32
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 672 {
      affine.store %cst, %arg2[%arg3] : memref<?xf32>
    }
    affine.for %arg3 = 0 to 2 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = arith.muli %0, %c336_i32 : i32
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
          affine.for %arg6 = 0 to 4 {
            %2 = affine.load %arg1[%arg3 * 36 + %arg6 + %arg4 * 12 + %arg5 * 4] : memref<?xi32>
            %3 = arith.addi %1, %2 : i32
            %4 = arith.index_cast %3 : i32 to index
            %5 = affine.load %arg0[%arg3 * 36 + %arg6 + %arg4 * 12 + %arg5 * 4] : memref<?xf32>
            %6 = memref.load %arg2[%4] : memref<?xf32>
            %7 = arith.addf %6, %5 : f32
            memref.store %7, %arg2[%4] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}
