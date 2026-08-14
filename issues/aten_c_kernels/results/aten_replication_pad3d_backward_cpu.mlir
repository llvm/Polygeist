#set = affine_set<(d0) : (-d0 + 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c-2_i32 = arith.constant -2 : i32
    %false = arith.constant false
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    affine.for %arg2 = 0 to 240 {
      affine.store %cst, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 8 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.addi %2, %c-2_i32 : i32
        %4 = arith.cmpi slt, %3, %c0_i32 : i32
        %5 = arith.select %4, %c0_i32, %3 : i32
        %6 = affine.if #set(%arg3) -> i1 {
          affine.yield %false : i1
        } else {
          %10 = arith.cmpi sge, %3, %c4_i32 : i32
          affine.yield %10 : i1
        }
        %7 = arith.select %6, %c3_i32, %5 : i32
        %8 = arith.addi %1, %7 : i32
        %9 = arith.muli %8, %c5_i32 : i32
        affine.for %arg4 = 0 to 9 {
          %10 = arith.index_cast %arg4 : index to i32
          %11 = arith.addi %10, %c-2_i32 : i32
          %12 = arith.cmpi slt, %11, %c0_i32 : i32
          %13 = arith.select %12, %c0_i32, %11 : i32
          %14 = affine.if #set(%arg4) -> i1 {
            affine.yield %false : i1
          } else {
            %18 = arith.cmpi sge, %11, %c5_i32 : i32
            affine.yield %18 : i1
          }
          %15 = arith.select %14, %c4_i32, %13 : i32
          %16 = arith.addi %9, %15 : i32
          %17 = arith.muli %16, %c6_i32 : i32
          affine.for %arg5 = 0 to 10 {
            %18 = arith.index_cast %arg5 : index to i32
            %19 = arith.addi %18, %c-2_i32 : i32
            %20 = arith.cmpi slt, %19, %c0_i32 : i32
            %21 = arith.select %20, %c0_i32, %19 : i32
            %22 = affine.if #set(%arg5) -> i1 {
              affine.yield %false : i1
            } else {
              %29 = arith.cmpi sge, %19, %c6_i32 : i32
              affine.yield %29 : i1
            }
            %23 = arith.select %22, %c5_i32, %21 : i32
            %24 = arith.addi %17, %23 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.load %arg0[%arg2 * 720 + %arg5 + %arg3 * 90 + %arg4 * 10] : memref<?xf32>
            %27 = memref.load %arg1[%25] : memref<?xf32>
            %28 = arith.addf %27, %26 : f32
            memref.store %28, %arg1[%25] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}
