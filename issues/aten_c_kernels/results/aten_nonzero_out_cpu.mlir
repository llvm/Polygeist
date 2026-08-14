module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nonzero_out_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %arg5) -> (i32) {
        %3 = arith.index_cast %arg6 : index to i32
        %4 = affine.load %arg0[%arg4, %arg6] : memref<?x64xf32>
        %5 = arith.cmpf une, %4, %cst : f32
        %6 = scf.if %5 -> (i32) {
          %7 = arith.index_cast %arg7 : i32 to index
          memref.store %1, %arg1[%7] : memref<?xi32>
          %8 = arith.addi %arg7, %c1_i32 : i32
          memref.store %3, %arg2[%7] : memref<?xi32>
          scf.yield %8 : i32
        } else {
          scf.yield %arg7 : i32
        }
        affine.yield %6 : i32
      }
      affine.yield %2 : i32
    }
    affine.store %0, %arg3[0] : memref<?xi32>
    return
  }
}
