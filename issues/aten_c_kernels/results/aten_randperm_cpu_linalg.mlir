#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_randperm_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c256 = arith.constant 256 : index
    %c1_i32 = arith.constant 1 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xi32>) {
    ^bb0(%out: i32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i32
      linalg.yield %1 : i32
    }
    affine.for %arg2 = 1 to 256 {
      %0 = arith.subi %c256, %arg2 : index
      %1 = arith.index_cast %0 : index to i32
      %2 = affine.load %arg0[-%arg2 + 256] : memref<?xi32>
      %3 = arith.addi %1, %c1_i32 : i32
      %4 = arith.remui %2, %3 : i32
      %5 = affine.load %arg1[-%arg2 + 256] : memref<?xi32>
      %6 = arith.index_cast %4 : i32 to index
      %7 = memref.load %arg1[%6] : memref<?xi32>
      affine.store %7, %arg1[-%arg2 + 256] : memref<?xi32>
      memref.store %5, %arg1[%6] : memref<?xi32>
    }
    return
  }
}

