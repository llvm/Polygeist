module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sobol_draw_cpu(%arg0: memref<?xi32>, %arg1: memref<?x32xi32>, %arg2: memref<?x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 2.32830644E-10 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?x8xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x32xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xi32>
    %3:2 = affine.for %arg3 = 0 to 256 iter_args(%arg4 = %2, %arg5 = %0) -> (tensor<?xi32>, tensor<?x8xf32>) {
      %6 = arith.index_cast %arg3 : index to i32
      %7:2 = scf.while (%arg6 = %6, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %10 = arith.andi %arg6, %c1_i32 : i32
        %11 = arith.cmpi ne, %10, %c0_i32 : i32
        scf.condition(%11) %arg7, %arg6 : i32, i32
      } do {
      ^bb0(%arg6: i32, %arg7: i32):
        %10 = arith.addi %arg6, %c1_i32 : i32
        %11 = arith.shrsi %arg7, %c1_i32 : i32
        scf.yield %11, %10 : i32, i32
      }
      %8 = arith.index_cast %7#0 : i32 to index
      %9:2 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?xi32>, tensor<?x8xf32>) {
        %extracted = tensor.extract %1[%arg6, %8] : tensor<?x32xi32>
        %extracted_0 = tensor.extract %arg7[%arg6] : tensor<?xi32>
        %10 = arith.xori %extracted_0, %extracted : i32
        %inserted = tensor.insert %10 into %arg7[%arg6] : tensor<?xi32>
        %11 = arith.uitofp %10 : i32 to f32
        %12 = arith.mulf %11, %cst : f32
        %inserted_1 = tensor.insert %12 into %arg8[%arg3, %arg6] : tensor<?x8xf32>
        affine.yield %inserted, %inserted_1 : tensor<?xi32>, tensor<?x8xf32>
      }
      affine.yield %9#0, %9#1 : tensor<?xi32>, tensor<?x8xf32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?x8xf32>
    memref.copy %4, %arg2 : memref<?x8xf32> to memref<?x8xf32>
    %5 = bufferization.to_memref %3#0 : memref<?xi32>
    memref.copy %5, %arg0 : memref<?xi32> to memref<?xi32>
    return
  }
}

