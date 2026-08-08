module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gcd_i32(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 4096 {
      %0 = affine.load %arg0[%arg3] : memref<?xi32>
      %1 = arith.cmpi slt, %0, %c0_i32 : i32
      %2 = scf.if %1 -> (i32) {
        %7 = arith.subi %c0_i32, %0 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %0 : i32
      }
      %3 = affine.load %arg1[%arg3] : memref<?xi32>
      %4 = arith.cmpi slt, %3, %c0_i32 : i32
      %5 = scf.if %4 -> (i32) {
        %7 = arith.subi %c0_i32, %3 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %3 : i32
      }
      %6:2 = scf.while (%arg4 = %5, %arg5 = %2) : (i32, i32) -> (i32, i32) {
        %7 = arith.cmpi ne, %arg4, %c0_i32 : i32
        scf.condition(%7) %arg5, %arg4 : i32, i32
      } do {
      ^bb0(%arg4: i32, %arg5: i32):
        %7 = arith.remsi %arg4, %arg5 : i32
        scf.yield %7, %arg5 : i32, i32
      }
      affine.store %6#0, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}
