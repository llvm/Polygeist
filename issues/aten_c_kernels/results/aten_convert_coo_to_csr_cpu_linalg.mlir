module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_convert_coo_to_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 65 {
      %0 = affine.load %alloca[] : memref<i32>
      %1 = arith.index_cast %arg2 : index to i32
      %2 = scf.while (%arg3 = %0) : (i32) -> i32 {
        %3 = arith.cmpi slt, %arg3, %c512_i32 : i32
        %4 = arith.index_cast %arg3 : i32 to index
        %5 = memref.load %arg0[%4] : memref<?xi32>
        %6 = arith.cmpi slt, %5, %1 : i32
        %7 = arith.addi %arg3, %c1_i32 : i32
        %8 = arith.select %6, %7, %arg3 : i32
        %9 = arith.select %3, %6, %false : i1
        %10 = arith.select %3, %8, %arg3 : i32
        scf.condition(%9) %10 : i32
      } do {
      ^bb0(%arg3: i32):
        scf.yield %arg3 : i32
      }
      affine.store %2, %arg1[%arg2] : memref<?xi32>
      affine.store %2, %alloca[] : memref<i32>
    }
    return
  }
}

