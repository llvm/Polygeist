#set = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 - 9 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_reflection_pad2d(%arg0: memref<?x8x8xf32>, %arg1: memref<?x10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 10 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = affine.if #set(%arg3) -> i32 {
          affine.yield %c1_i32 : i32
        } else {
          %3 = affine.if #set1(%arg3) -> i32 {
            affine.yield %c6_i32 : i32
          } else {
            %4 = arith.addi %0, %c-1_i32 : i32
            affine.yield %4 : i32
          }
          affine.yield %3 : i32
        }
        %2 = arith.index_cast %1 : i32 to index
        affine.for %arg4 = 0 to 10 {
          %3 = arith.index_cast %arg4 : index to i32
          %4 = affine.if #set(%arg4) -> i32 {
            affine.yield %c1_i32 : i32
          } else {
            %7 = affine.if #set1(%arg4) -> i32 {
              affine.yield %c6_i32 : i32
            } else {
              %8 = arith.addi %3, %c-1_i32 : i32
              affine.yield %8 : i32
            }
            affine.yield %7 : i32
          }
          %5 = arith.index_cast %4 : i32 to index
          %6 = memref.load %arg0[%arg2, %2, %5] : memref<?x8x8xf32>
          affine.store %6, %arg1[%arg2, %arg3, %arg4] : memref<?x10x10xf32>
        }
      }
    }
    return
  }
}
