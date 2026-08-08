module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_jagged_to_padded_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 64 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = affine.load %arg1[%arg3] : memref<?xi32>
        %2 = arith.addi %1, %0 : i32
        %3 = affine.load %arg1[%arg3 + 1] : memref<?xi32>
        %4 = arith.cmpi slt, %2, %3 : i32
        %5 = scf.if %4 -> (f32) {
          %6 = arith.index_cast %2 : i32 to index
          %7 = memref.load %arg0[%6] : memref<?xf32>
          scf.yield %7 : f32
        } else {
          scf.yield %cst : f32
        }
        affine.store %5, %arg2[%arg3, %arg4] : memref<?x64xf32>
      }
    }
    return
  }
}
