module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool3d_legacy_backward_cpu(%arg0: memref<?x3x4x5xf32>, %arg1: memref<?x3x4x5xi32>, %arg2: memref<?x8x9x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c90_i32 = arith.constant 90 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?x3x4x5xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?x3x4x5xf32>
    %2 = "polygeist.memref2pointer"(%arg2) : (memref<?x8x9x10xf32>) -> !llvm.ptr
    affine.for %arg3 = 0 to 1440 {
      %3 = arith.index_cast %arg3 : index to i32
      %4 = llvm.getelementptr %2[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %4 : f32, !llvm.ptr
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 4 {
          affine.for %arg6 = 0 to 5 {
            %extracted = tensor.extract %0[%arg3, %arg4, %arg5, %arg6] : tensor<?x3x4x5xi32>
            %3 = arith.divsi %extracted, %c90_i32 : i32
            %4 = arith.index_cast %3 : i32 to index
            %5 = arith.divsi %extracted, %c10_i32 : i32
            %6 = arith.remsi %5, %c9_i32 : i32
            %7 = arith.index_cast %6 : i32 to index
            %8 = arith.remsi %extracted, %c10_i32 : i32
            %9 = arith.index_cast %8 : i32 to index
            %extracted_0 = tensor.extract %1[%arg3, %arg4, %arg5, %arg6] : tensor<?x3x4x5xf32>
            %10 = memref.load %arg2[%arg3, %4, %7, %9] : memref<?x8x9x10xf32>
            %11 = arith.addf %10, %extracted_0 : f32
            memref.store %11, %arg2[%arg3, %4, %7, %9] : memref<?x8x9x10xf32>
          }
        }
      }
    }
    return
  }
}

