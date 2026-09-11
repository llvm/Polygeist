module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_poisson_transform_cpu(%arg0: memref<?xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %0) -> (tensor<?xi32>) {
      %5:2 = scf.while (%arg5 = %c0_i32, %arg6 = %cst) : (i32, f32) -> (i32, f32) {
        %7 = arith.cmpi slt, %arg5, %c64_i32 : i32
        %extracted = tensor.extract %2[%arg3] : tensor<?xf32>
        %8 = arith.negf %extracted : f32
        %9 = math.exp %8 : f32
        %10 = arith.cmpf ogt, %arg6, %9 : f32
        %11 = arith.addi %arg5, %c1_i32 : i32
        %12 = arith.index_cast %arg5 : i32 to index
        %extracted_0 = tensor.extract %1[%arg3, %12] : tensor<?x64xf32>
        %13 = arith.mulf %arg6, %extracted_0 : f32
        %14 = arith.select %10, %11, %arg5 : i32
        %15 = arith.select %10, %13, %arg6 : f32
        %16 = arith.select %7, %10, %false : i1
        %17 = arith.select %7, %14, %arg5 : i32
        %18 = arith.select %7, %15, %arg6 : f32
        scf.condition(%16) %17, %18 : i32, f32
      } do {
      ^bb0(%arg5: i32, %arg6: f32):
        scf.yield %arg5, %arg6 : i32, f32
      }
      %6 = arith.addi %5#0, %c-1_i32 : i32
      %inserted = tensor.insert %6 into %arg4[%arg3] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

