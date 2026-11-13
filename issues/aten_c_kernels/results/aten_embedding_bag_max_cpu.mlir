module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_embedding_bag_max_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x16xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 64 {
        %0 = affine.load %arg1[%arg3, 0] : memref<?x16xi32>
        %1 = arith.index_cast %0 : i32 to index
        %2 = memref.load %arg0[%1, %arg4] : memref<?x64xf32>
        %3 = affine.for %arg5 = 1 to 16 iter_args(%arg6 = %2) -> (f32) {
          %4 = affine.load %arg1[%arg3, %arg5] : memref<?x16xi32>
          %5 = arith.index_cast %4 : i32 to index
          %6 = memref.load %arg0[%5, %arg4] : memref<?x64xf32>
          %7 = arith.cmpf ogt, %6, %arg6 : f32
          %8 = scf.if %7 -> (f32) {
            %9 = memref.load %arg0[%5, %arg4] : memref<?x64xf32>
            scf.yield %9 : f32
          } else {
            scf.yield %arg6 : f32
          }
          affine.yield %8 : f32
        }
        affine.store %3, %arg2[%arg3, %arg4] : memref<?x64xf32>
      }
    }
    return
  }
}
