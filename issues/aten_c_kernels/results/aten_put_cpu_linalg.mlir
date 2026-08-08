module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_put_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg3, %c0_i32 : i32
    affine.for %arg4 = 0 to 32 {
      scf.if %0 {
        %1 = affine.load %arg1[%arg4] : memref<?xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = affine.load %arg2[%arg4] : memref<?xf32>
        %4 = memref.load %arg0[%2] : memref<?xf32>
        %5 = arith.addf %4, %3 : f32
        memref.store %5, %arg0[%2] : memref<?xf32>
      } else {
        %1 = affine.load %arg1[%arg4] : memref<?xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = affine.load %arg2[%arg4] : memref<?xf32>
        memref.store %3, %arg0[%2] : memref<?xf32>
      }
    }
    return
  }
}

