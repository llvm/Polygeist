#set = affine_set<(d0) : (-d0 + 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_reflection_pad2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-2_i32 = arith.constant -2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 8 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.addi %2, %c-2_i32 : i32
        %4 = affine.if #set(%arg3) -> i32 {
          %9 = arith.subi %c2_i32, %2 : i32
          affine.yield %9 : i32
        } else {
          affine.yield %3 : i32
        }
        %5 = arith.cmpi sge, %4, %c4_i32 : i32
        %6 = scf.if %5 -> (i32) {
          %9 = arith.subi %c6_i32, %4 : i32
          scf.yield %9 : i32
        } else {
          scf.yield %4 : i32
        }
        %7 = arith.addi %1, %6 : i32
        %8 = arith.muli %7, %c5_i32 : i32
        affine.for %arg4 = 0 to 9 {
          %9 = arith.index_cast %arg4 : index to i32
          %10 = arith.addi %9, %c-2_i32 : i32
          %11 = affine.if #set(%arg4) -> i32 {
            %17 = arith.subi %c2_i32, %9 : i32
            affine.yield %17 : i32
          } else {
            affine.yield %10 : i32
          }
          %12 = arith.cmpi sge, %11, %c5_i32 : i32
          %13 = scf.if %12 -> (i32) {
            %17 = arith.subi %c8_i32, %11 : i32
            scf.yield %17 : i32
          } else {
            scf.yield %11 : i32
          }
          %14 = arith.addi %8, %13 : i32
          %15 = arith.index_cast %14 : i32 to index
          %16 = memref.load %arg0[%15] : memref<?xf32>
          affine.store %16, %arg1[%arg4 + %arg2 * 72 + %arg3 * 9] : memref<?xf32>
        }
      }
    }
    return
  }
}
