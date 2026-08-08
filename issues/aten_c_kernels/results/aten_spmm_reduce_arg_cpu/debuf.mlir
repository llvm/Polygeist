#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_arg_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: i32, %arg5: memref<?x24xf32>, %arg6: memref<?x24xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x24xi32>
    %1 = bufferization.to_tensor %arg5 : memref<?x24xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?x24xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?xf32>
    %4 = bufferization.to_tensor %arg1 : memref<?xi32>
    %5 = bufferization.to_tensor %arg0 : memref<?xi32>
    %6 = arith.cmpi ne, %arg4, %c0_i32 : i32
    %7 = arith.select %6, %cst, %cst_0 : f32
    %8:2 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %1, %arg9 = %0) -> (tensor<?x24xf32>, tensor<?x24xi32>) {
      %11:2 = affine.for %arg10 = 0 to 24 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<?x24xf32>, tensor<?x24xi32>) {
        %extracted = tensor.extract %5[%arg7] : tensor<?xi32>
        %12 = affine.apply #map(%arg7)
        %extracted_1 = tensor.extract %5[%12] : tensor<?xi32>
        %13 = arith.index_cast %extracted_1 : i32 to index
        %14 = arith.index_cast %extracted : i32 to index
        %15:2 = scf.for %arg13 = %14 to %13 step %c1 iter_args(%arg14 = %c-1_i32, %arg15 = %7) -> (i32, f32) {
          %16 = arith.index_cast %arg13 : index to i32
          %extracted_3 = tensor.extract %3[%arg13] : tensor<?xf32>
          %extracted_4 = tensor.extract %4[%arg13] : tensor<?xi32>
          %17 = arith.index_cast %extracted_4 : i32 to index
          %extracted_5 = tensor.extract %2[%17, %arg10] : tensor<?x24xf32>
          %18 = arith.mulf %extracted_3, %extracted_5 : f32
          %19 = arith.cmpf ogt, %18, %arg15 : f32
          %20 = arith.select %6, %19, %false : i1
          %21 = arith.cmpf olt, %18, %arg15 : f32
          %22 = arith.select %6, %false, %21 : i1
          %23 = arith.select %20, %true, %22 : i1
          %24 = arith.select %23, %16, %arg14 : i32
          %25 = arith.select %23, %18, %arg15 : f32
          scf.yield %24, %25 : i32, f32
        }
        %inserted = tensor.insert %15#1 into %arg11[%arg7, %arg10] : tensor<?x24xf32>
        %inserted_2 = tensor.insert %15#0 into %arg12[%arg7, %arg10] : tensor<?x24xi32>
        affine.yield %inserted, %inserted_2 : tensor<?x24xf32>, tensor<?x24xi32>
      }
      affine.yield %11#0, %11#1 : tensor<?x24xf32>, tensor<?x24xi32>
    }
    %9 = bufferization.to_memref %8#1 : memref<?x24xi32>
    memref.copy %9, %arg6 : memref<?x24xi32> to memref<?x24xi32>
    %10 = bufferization.to_memref %8#0 : memref<?x24xf32>
    memref.copy %10, %arg5 : memref<?x24xf32> to memref<?x24xf32>
    return
  }
}

