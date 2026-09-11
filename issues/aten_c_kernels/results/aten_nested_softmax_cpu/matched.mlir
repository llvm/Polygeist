module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_softmax_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %0) -> (tensor<?x64xf32>) {
      %5:3 = scf.while (%arg5 = %c0_i32, %arg6 = %cst, %arg7 = %arg4) : (i32, f32, tensor<?x64xf32>) -> (f32, i32, tensor<?x64xf32>) {
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %7 = arith.cmpi slt, %arg5, %extracted : i32
        scf.condition(%7) %arg6, %arg5, %arg7 : f32, i32, tensor<?x64xf32>
      } do {
      ^bb0(%arg5: f32, %arg6: i32, %arg7: tensor<?x64xf32>):
        %7 = arith.index_cast %arg6 : i32 to index
        %extracted = tensor.extract %2[%arg3, %7] : tensor<?x64xf32>
        %8 = math.exp %extracted : f32
        %inserted = tensor.insert %8 into %arg7[%arg3, %7] : tensor<?x64xf32>
        %extracted_0 = tensor.extract %inserted[%arg3, %7] : tensor<?x64xf32>
        %9 = arith.addf %arg5, %extracted_0 : f32
        %10 = arith.addi %arg6, %c1_i32 : i32
        scf.yield %10, %9, %inserted : i32, f32, tensor<?x64xf32>
      }
      %6:2 = scf.while (%arg5 = %c0_i32, %arg6 = %5#2) : (i32, tensor<?x64xf32>) -> (i32, tensor<?x64xf32>) {
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %7 = arith.cmpi slt, %arg5, %extracted : i32
        scf.condition(%7) %arg5, %arg6 : i32, tensor<?x64xf32>
      } do {
      ^bb0(%arg5: i32, %arg6: tensor<?x64xf32>):
        %7 = arith.index_cast %arg5 : i32 to index
        %extracted = tensor.extract %arg6[%arg3, %7] : tensor<?x64xf32>
        %8 = arith.divf %extracted, %5#0 : f32
        %inserted = tensor.insert %8 into %arg6[%arg3, %7] : tensor<?x64xf32>
        %9 = arith.addi %arg5, %c1_i32 : i32
        scf.yield %9, %inserted : i32, tensor<?x64xf32>
      }
      affine.yield %6#1 : tensor<?x64xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x64xf32>
    memref.copy %4, %arg2 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

