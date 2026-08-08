#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_pdist_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?xf32>, %arg2: memref<?x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %alloca = memref.alloca() : memref<120xf32>
    affine.for %arg3 = 0 to 16 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = arith.subi %c31_i32, %0 : i32
      %2 = arith.muli %0, %1 : i32
      %3 = arith.divsi %2, %c2_i32 : i32
      affine.for %arg4 = #map(%arg3) to 16 {
        %4 = arith.index_cast %arg4 : index to i32
        %5 = arith.subi %4, %0 : i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.addi %3, %6 : i32
        %8 = arith.index_cast %7 : i32 to index
        memref.store %cst, %alloca[%8] : memref<120xf32>
        %9 = affine.for %arg5 = 0 to 32 iter_args(%arg6 = %cst) -> (f32) {
          %12 = affine.load %arg0[%arg3, %arg5] : memref<?x32xf32>
          %13 = affine.load %arg0[%arg4, %arg5] : memref<?x32xf32>
          %14 = arith.subf %12, %13 : f32
          %15 = arith.mulf %14, %14 : f32
          %16 = arith.addf %arg6, %15 : f32
          memref.store %16, %alloca[%8] : memref<120xf32>
          affine.yield %16 : f32
        }
        %10 = memref.load %alloca[%8] : memref<120xf32>
        %11 = math.sqrt %10 : f32
        memref.store %11, %alloca[%8] : memref<120xf32>
      }
    }
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 32 {
        affine.store %cst, %arg2[%arg3, %arg4] : memref<?x32xf32>
      }
    }
    affine.for %arg3 = 0 to 16 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = arith.subi %c31_i32, %0 : i32
      %2 = arith.muli %0, %1 : i32
      %3 = arith.divsi %2, %c2_i32 : i32
      affine.for %arg4 = #map(%arg3) to 16 {
        %4 = arith.index_cast %arg4 : index to i32
        %5 = arith.subi %4, %0 : i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.addi %3, %6 : i32
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %alloca[%8] : memref<120xf32>
        %10 = arith.cmpf oeq, %9, %cst : f32
        %11 = scf.if %10 -> (f32) {
          scf.yield %cst : f32
        } else {
          %12 = memref.load %arg1[%8] : memref<?xf32>
          %13 = arith.divf %12, %9 : f32
          scf.yield %13 : f32
        }
        affine.for %arg5 = 0 to 32 {
          %12 = affine.load %arg0[%arg3, %arg5] : memref<?x32xf32>
          %13 = affine.load %arg0[%arg4, %arg5] : memref<?x32xf32>
          %14 = arith.subf %12, %13 : f32
          %15 = arith.mulf %11, %14 : f32
          %16 = affine.load %arg2[%arg3, %arg5] : memref<?x32xf32>
          %17 = arith.addf %16, %15 : f32
          affine.store %17, %arg2[%arg3, %arg5] : memref<?x32xf32>
          %18 = affine.load %arg2[%arg4, %arg5] : memref<?x32xf32>
          %19 = arith.subf %18, %15 : f32
          affine.store %19, %arg2[%arg4, %arg5] : memref<?x32xf32>
        }
      }
    }
    return
  }
}
