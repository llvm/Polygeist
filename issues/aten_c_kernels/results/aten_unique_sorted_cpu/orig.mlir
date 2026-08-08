#map = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0) : (d0 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_sorted_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = #map(%arg3) to 1024 {
        %1 = affine.load %arg0[%arg4] : memref<?xi32>
        %2 = affine.load %arg0[%arg3] : memref<?xi32>
        %3 = arith.cmpi slt, %1, %2 : i32
        scf.if %3 {
          affine.store %1, %arg0[%arg3] : memref<?xi32>
          affine.store %2, %arg0[%arg4] : memref<?xi32>
        }
      }
    }
    %0 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %c0_i32) -> (i32) {
      %1 = affine.if #set(%arg3) -> i1 {
        affine.yield %true : i1
      } else {
        %3 = affine.load %arg0[%arg3] : memref<?xi32>
        %4 = affine.load %arg0[%arg3 - 1] : memref<?xi32>
        %5 = arith.cmpi ne, %3, %4 : i32
        affine.yield %5 : i1
      }
      %2 = scf.if %1 -> (i32) {
        %3 = arith.addi %arg4, %c1_i32 : i32
        %4 = arith.index_cast %arg4 : i32 to index
        %5 = affine.load %arg0[%arg3] : memref<?xi32>
        memref.store %5, %arg1[%4] : memref<?xi32>
        scf.yield %3 : i32
      } else {
        scf.yield %arg4 : i32
      }
      affine.yield %2 : i32
    }
    affine.store %0, %arg2[0] : memref<?xi32>
    return
  }
}
