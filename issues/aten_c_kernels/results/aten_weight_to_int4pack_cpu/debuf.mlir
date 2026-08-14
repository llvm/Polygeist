#map = affine_map<(d0, d1) -> (d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_weight_to_int4pack_cpu(%arg0: memref<?x64xi8>, %arg1: memref<?x32xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x32xi8>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xi8>
    %2 = affine.for %arg2 = 0 to 48 iter_args(%arg3 = %0) -> (tensor<?x32xi8>) {
      %4 = affine.for %arg4 = 0 to 64 step 2 iter_args(%arg5 = %arg3) -> (tensor<?x32xi8>) {
        %extracted = tensor.extract %1[%arg2, %arg4] : tensor<?x64xi8>
        %5 = arith.extui %extracted : i8 to i32
        %6 = arith.andi %5, %c15_i32 : i32
        %7 = affine.apply #map(%arg2, %arg4)
        %extracted_0 = tensor.extract %1[%arg2, %7] : tensor<?x64xi8>
        %8 = arith.extui %extracted_0 : i8 to i32
        %9 = arith.andi %8, %c15_i32 : i32
        %10 = arith.shli %9, %c4_i32 : i32
        %11 = arith.ori %6, %10 : i32
        %12 = arith.trunci %11 : i32 to i8
        %13 = arith.cmpi slt, %arg4, %c0 : index
        %14 = arith.subi %c-1, %arg4 : index
        %15 = arith.select %13, %14, %arg4 : index
        %16 = arith.divsi %15, %c2 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %inserted = tensor.insert %12 into %arg5[%arg2, %18] : tensor<?x32xi8>
        affine.yield %inserted : tensor<?x32xi8>
      }
      affine.yield %4 : tensor<?x32xi8>
    }
    %3 = bufferization.to_memref %2 : memref<?x32xi8>
    memref.copy %3, %arg1 : memref<?x32xi8> to memref<?x32xi8>
    return
  }
}

