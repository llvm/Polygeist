module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_convert_coo_to_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg2 = 0 to 65 iter_args(%arg3 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = scf.while (%arg4 = %arg3) : (i32) -> i32 {
        %3 = arith.cmpi slt, %arg4, %c512_i32 : i32
        %4:2 = scf.if %3 -> (i1, i32) {
          %5 = arith.index_cast %arg4 : i32 to index
          %6 = memref.load %arg0[%5] : memref<?xi32>
          %7 = arith.cmpi slt, %6, %1 : i32
          %8 = scf.if %7 -> (i32) {
            %9 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %9 : i32
          } else {
            scf.yield %arg4 : i32
          }
          scf.yield %7, %8 : i1, i32
        } else {
          scf.yield %false, %arg4 : i1, i32
        }
        scf.condition(%4#0) %4#1 : i32
      } do {
      ^bb0(%arg4: i32):
        scf.yield %arg4 : i32
      }
      affine.store %2, %arg1[%arg2] : memref<?xi32>
      affine.yield %2 : i32
    }
    return
  }
}
