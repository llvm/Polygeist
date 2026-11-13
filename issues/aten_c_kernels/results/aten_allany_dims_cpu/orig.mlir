module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_allany_dims_cpu(%arg0: memref<?x64xi32>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %c0_i32 : i32
    affine.for %arg3 = 0 to 32 {
      %1 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg1) -> (i32) {
        %2 = scf.if %0 -> (i32) {
          %3 = arith.cmpi ne, %arg5, %c0_i32 : i32
          %4 = scf.if %3 -> (i1) {
            %6 = affine.load %arg0[%arg3, %arg4] : memref<?x64xi32>
            %7 = arith.cmpi ne, %6, %c0_i32 : i32
            scf.yield %7 : i1
          } else {
            scf.yield %false : i1
          }
          %5 = arith.extsi %4 : i1 to i32
          scf.yield %5 : i32
        } else {
          %3 = arith.cmpi ne, %arg5, %c0_i32 : i32
          %4 = scf.if %3 -> (i1) {
            scf.yield %true : i1
          } else {
            %6 = affine.load %arg0[%arg3, %arg4] : memref<?x64xi32>
            %7 = arith.cmpi ne, %6, %c0_i32 : i32
            scf.yield %7 : i1
          }
          %5 = arith.extsi %4 : i1 to i32
          scf.yield %5 : i32
        }
        affine.yield %2 : i32
      }
      affine.store %1, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}
