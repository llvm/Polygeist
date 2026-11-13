module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_weight_to_int4pack_cpu(%arg0: memref<?x64xi8>, %arg1: memref<?x32xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c4_i32 = arith.constant 4 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    affine.for %arg2 = 0 to 48 {
      affine.for %arg3 = 0 to 64 step 2 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<?x64xi8>
        %1 = arith.extui %0 : i8 to i32
        %2 = arith.andi %1, %c15_i32 : i32
        %3 = affine.load %arg0[%arg2, %arg3 + 1] : memref<?x64xi8>
        %4 = arith.extui %3 : i8 to i32
        %5 = arith.andi %4, %c15_i32 : i32
        %6 = arith.shli %5, %c4_i32 : i32
        %7 = arith.ori %2, %6 : i32
        %8 = arith.trunci %7 : i32 to i8
        %9 = arith.cmpi slt, %arg3, %c0 : index
        %10 = arith.subi %c-1, %arg3 : index
        %11 = arith.select %9, %10, %arg3 : index
        %12 = arith.divsi %11, %c2 : index
        %13 = arith.subi %c-1, %12 : index
        %14 = arith.select %9, %13, %12 : index
        memref.store %8, %arg1[%arg2, %14] : memref<?x32xi8>
      }
    }
    return
  }
}

