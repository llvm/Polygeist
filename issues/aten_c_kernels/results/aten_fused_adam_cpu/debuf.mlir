module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_adam_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32, %arg12: f32, %arg13: i32, %arg14: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?xf32>
    %5 = arith.divf %arg5, %arg8 : f32
    %6 = arith.cmpi ne, %arg13, %c0_i32 : i32
    %7 = arith.cmpf une, %arg10, %cst : f32
    %8 = arith.subf %cst_0, %arg6 : f32
    %9 = arith.subf %cst_0, %arg7 : f32
    %10 = arith.cmpi ne, %arg14, %c0_i32 : i32
    %11:5 = affine.for %arg15 = 0 to 4096 iter_args(%arg16 = %4, %arg17 = %3, %arg18 = %2, %arg19 = %1, %arg20 = %0) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg17[%arg15] : tensor<?xf32>
      %17 = arith.divf %extracted, %arg12 : f32
      %inserted = tensor.insert %17 into %arg17[%arg15] : tensor<?xf32>
      %18 = arith.negf %17 : f32
      %19 = arith.select %6, %18, %17 : f32
      %extracted_1 = tensor.extract %arg16[%arg15] : tensor<?xf32>
      %20 = arith.mulf %extracted_1, %arg10 : f32
      %21 = arith.addf %19, %20 : f32
      %22 = arith.select %7, %21, %19 : f32
      %extracted_2 = tensor.extract %arg18[%arg15] : tensor<?xf32>
      %23 = arith.subf %22, %extracted_2 : f32
      %24 = arith.mulf %8, %23 : f32
      %25 = arith.addf %extracted_2, %24 : f32
      %inserted_3 = tensor.insert %25 into %arg18[%arg15] : tensor<?xf32>
      %extracted_4 = tensor.extract %arg19[%arg15] : tensor<?xf32>
      %26 = arith.mulf %arg7, %extracted_4 : f32
      %27 = arith.mulf %9, %22 : f32
      %28 = arith.mulf %27, %22 : f32
      %29 = arith.addf %26, %28 : f32
      %inserted_5 = tensor.insert %29 into %arg19[%arg15] : tensor<?xf32>
      %30:2 = scf.if %10 -> (f32, tensor<?xf32>) {
        %extracted_9 = tensor.extract %arg20[%arg15] : tensor<?xf32>
        %37 = arith.cmpf ogt, %extracted_9, %29 : f32
        %38 = arith.select %37, %extracted_9, %29 : f32
        %inserted_10 = tensor.insert %38 into %arg20[%arg15] : tensor<?xf32>
        scf.yield %38, %inserted_10 : f32, tensor<?xf32>
      } else {
        scf.yield %29, %arg20 : f32, tensor<?xf32>
      }
      %extracted_6 = tensor.extract %inserted_3[%arg15] : tensor<?xf32>
      %31 = arith.mulf %5, %extracted_6 : f32
      %32 = math.sqrt %30#0 : f32
      %33 = arith.divf %32, %arg9 : f32
      %34 = arith.addf %33, %arg11 : f32
      %35 = arith.divf %31, %34 : f32
      %extracted_7 = tensor.extract %arg16[%arg15] : tensor<?xf32>
      %36 = arith.subf %extracted_7, %35 : f32
      %inserted_8 = tensor.insert %36 into %arg16[%arg15] : tensor<?xf32>
      affine.yield %inserted_8, %inserted, %inserted_3, %inserted_5, %30#1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %12 = bufferization.to_memref %11#4 : memref<?xf32>
    memref.copy %12, %arg4 : memref<?xf32> to memref<?xf32>
    %13 = bufferization.to_memref %11#3 : memref<?xf32>
    memref.copy %13, %arg3 : memref<?xf32> to memref<?xf32>
    %14 = bufferization.to_memref %11#2 : memref<?xf32>
    memref.copy %14, %arg2 : memref<?xf32> to memref<?xf32>
    %15 = bufferization.to_memref %11#1 : memref<?xf32>
    memref.copy %15, %arg1 : memref<?xf32> to memref<?xf32>
    %16 = bufferization.to_memref %11#0 : memref<?xf32>
    memref.copy %16, %arg0 : memref<?xf32> to memref<?xf32>
    return
  }
}

