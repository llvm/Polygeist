module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool3d_backward_cpu(%arg0: memref<?x2x3x4x5xf32>, %arg1: memref<?x2x3x4x5xi32>, %arg2: memref<?x2x8x9x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c90_i32 = arith.constant 90 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %0 = "polygeist.memref2pointer"(%arg2) : (memref<?x2x8x9x10xf32>) -> !llvm.ptr
    affine.for %arg3 = 0 to 1440 {
      %1 = arith.index_cast %arg3 : index to i32
      %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %2 : f32, !llvm.ptr
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 4 {
          affine.for %arg6 = 0 to 5 {
            %1 = affine.load %arg1[0, %arg3, %arg4, %arg5, %arg6] : memref<?x2x3x4x5xi32>
            %2 = arith.divsi %1, %c90_i32 : i32
            %3 = arith.index_cast %2 : i32 to index
            %4 = arith.divsi %1, %c10_i32 : i32
            %5 = arith.remsi %4, %c9_i32 : i32
            %6 = arith.index_cast %5 : i32 to index
            %7 = arith.remsi %1, %c10_i32 : i32
            %8 = arith.index_cast %7 : i32 to index
            %9 = affine.load %arg0[0, %arg3, %arg4, %arg5, %arg6] : memref<?x2x3x4x5xf32>
            %10 = memref.load %arg2[%c0, %arg3, %3, %6, %8] : memref<?x2x8x9x10xf32>
            %11 = arith.addf %10, %9 : f32
            memref.store %11, %arg2[%c0, %arg3, %3, %6, %8] : memref<?x2x8x9x10xf32>
          }
        }
      }
    }
    return
  }
}

