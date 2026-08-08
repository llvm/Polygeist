module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_compressed_block_convert_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x16x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    affine.for %arg2 = 0 to 64 {
      %0 = arith.cmpi slt, %arg2, %c0 : index
      %1 = arith.subi %c-1, %arg2 : index
      %2 = arith.select %0, %1, %arg2 : index
      %3 = arith.divsi %2, %c4 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg2, %c4 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c4 : index
      %9 = arith.select %7, %8, %6 : index
      affine.for %arg3 = 0 to 64 {
        %10 = affine.load %arg0[%arg2, %arg3] : memref<?x64xf32>
        %11 = arith.cmpi slt, %arg3, %c0 : index
        %12 = arith.subi %c-1, %arg3 : index
        %13 = arith.select %11, %12, %arg3 : index
        %14 = arith.divsi %13, %c4 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = arith.remsi %arg3, %c4 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c4 : index
        %20 = arith.select %18, %19, %17 : index
        memref.store %10, %arg1[%5, %16, %9, %20] : memref<?x16x4x4xf32>
      }
    }
    return
  }
}

