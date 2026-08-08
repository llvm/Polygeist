module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_sgd_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = arith.cmpi ne, %arg8, %c0_i32 : i32
    %4 = arith.cmpf une, %arg6, %cst : f32
    %5 = arith.cmpf une, %arg4, %cst : f32
    %6 = arith.cmpi ne, %arg9, %c0_i32 : i32
    %7 = arith.subf %cst_0, %arg5 : f32
    %8 = arith.cmpi ne, %arg10, %c0_i32 : i32
    %9:3 = affine.for %arg11 = 0 to 4096 iter_args(%arg12 = %2, %arg13 = %1, %arg14 = %0) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg13[%arg11] : tensor<?xf32>
      %13 = arith.divf %extracted, %arg7 : f32
      %inserted = tensor.insert %13 into %arg13[%arg11] : tensor<?xf32>
      %14 = arith.negf %13 : f32
      %15 = arith.select %3, %14, %13 : f32
      %extracted_1 = tensor.extract %arg12[%arg11] : tensor<?xf32>
      %16 = arith.mulf %extracted_1, %arg6 : f32
      %17 = arith.addf %15, %16 : f32
      %18 = arith.select %4, %17, %15 : f32
      %19:2 = scf.if %5 -> (f32, tensor<?xf32>) {
        %extracted_4 = tensor.extract %arg14[%arg11] : tensor<?xf32>
        %22 = arith.mulf %extracted_4, %arg4 : f32
        %23 = arith.mulf %18, %7 : f32
        %24 = arith.addf %22, %23 : f32
        %25 = arith.select %6, %18, %24 : f32
        %inserted_5 = tensor.insert %25 into %arg14[%arg11] : tensor<?xf32>
        %26 = arith.mulf %arg4, %25 : f32
        %27 = arith.addf %18, %26 : f32
        %28 = arith.select %8, %27, %25 : f32
        scf.yield %28, %inserted_5 : f32, tensor<?xf32>
      } else {
        scf.yield %18, %arg14 : f32, tensor<?xf32>
      }
      %20 = arith.mulf %arg3, %19#0 : f32
      %extracted_2 = tensor.extract %arg12[%arg11] : tensor<?xf32>
      %21 = arith.subf %extracted_2, %20 : f32
      %inserted_3 = tensor.insert %21 into %arg12[%arg11] : tensor<?xf32>
      affine.yield %inserted_3, %inserted, %19#1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %10 = bufferization.to_memref %9#2 : memref<?xf32>
    memref.copy %10, %arg2 : memref<?xf32> to memref<?xf32>
    %11 = bufferization.to_memref %9#1 : memref<?xf32>
    memref.copy %11, %arg1 : memref<?xf32> to memref<?xf32>
    %12 = bufferization.to_memref %9#0 : memref<?xf32>
    memref.copy %12, %arg0 : memref<?xf32> to memref<?xf32>
    return
  }
}

