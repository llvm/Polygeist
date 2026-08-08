#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 + d1 * 9 + d2 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c42 = arith.constant 42 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3:2 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?xf32>, tensor<?xi32>) {
      %6 = arith.muli %arg3, %c42 : index
      %7:2 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?xf32>, tensor<?xi32>) {
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %8 = bufferization.to_tensor %alloca : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        %9 = bufferization.to_tensor %alloca_0 : memref<?xf32>
        %10:4 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %8, %arg11 = %9, %arg12 = %arg7, %arg13 = %arg8) -> (tensor<?xi32>, tensor<?xf32>, tensor<?xf32>, tensor<?xi32>) {
          %11 = arith.muli %arg9, %c7 : index
          %12 = arith.cmpi slt, %11, %c0 : index
          %13 = arith.subi %c-1, %11 : index
          %14 = arith.select %12, %13, %11 : index
          %15 = arith.divsi %14, %c3 : index
          %16 = arith.subi %c-1, %15 : index
          %17 = arith.select %12, %16, %15 : index
          %18 = arith.addi %17, %c3 : index
          %inserted = tensor.insert %c0_i32 into %arg10[%arg9] : tensor<?xi32>
          %inserted_1 = tensor.insert %cst into %arg11[%arg9] : tensor<?xf32>
          %19:2 = affine.for %arg14 = #map(%arg6) to #map1(%arg6) iter_args(%arg15 = %inserted, %arg16 = %inserted_1) -> (tensor<?xi32>, tensor<?xf32>) {
            %extracted_5 = tensor.extract %arg15[%arg9] : tensor<?xi32>
            %extracted_6 = tensor.extract %arg16[%arg9] : tensor<?xf32>
            %22 = arith.index_cast %arg14 : index to i32
            %23 = arith.muli %22, %c7_i32 : i32
            %24 = arith.muli %arg14, %c7 : index
            %25:2 = scf.for %arg17 = %17 to %18 step %c1 iter_args(%arg18 = %extracted_5, %arg19 = %extracted_6) -> (i32, f32) {
              %26 = arith.index_cast %arg17 : index to i32
              %27 = arith.addi %arg17, %6 : index
              %28 = arith.addi %27, %24 : index
              %extracted_9 = tensor.extract %2[%28] : tensor<?xf32>
              %29 = arith.cmpf ogt, %extracted_9, %arg19 : f32
              %30 = arith.select %29, %extracted_9, %arg19 : f32
              %31 = arith.addi %23, %26 : i32
              %32 = arith.select %29, %31, %arg18 : i32
              scf.yield %32, %30 : i32, f32
            }
            %inserted_7 = tensor.insert %25#0 into %arg15[%arg9] : tensor<?xi32>
            %inserted_8 = tensor.insert %25#1 into %arg16[%arg9] : tensor<?xf32>
            affine.yield %inserted_7, %inserted_8 : tensor<?xi32>, tensor<?xf32>
          }
          %extracted = tensor.extract %19#0[%arg9] : tensor<?xi32>
          %extracted_2 = tensor.extract %19#1[%arg9] : tensor<?xf32>
          %20 = affine.apply #map2(%arg9, %arg3, %arg6)
          %inserted_3 = tensor.insert %extracted_2 into %arg12[%20] : tensor<?xf32>
          %21 = affine.apply #map2(%arg9, %arg3, %arg6)
          %inserted_4 = tensor.insert %extracted into %arg13[%21] : tensor<?xi32>
          affine.yield %19#0, %19#1, %inserted_3, %inserted_4 : tensor<?xi32>, tensor<?xf32>, tensor<?xf32>, tensor<?xi32>
        }
        affine.yield %10#2, %10#3 : tensor<?xf32>, tensor<?xi32>
      }
      affine.yield %7#0, %7#1 : tensor<?xf32>, tensor<?xi32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    %5 = bufferization.to_memref %3#0 : memref<?xf32>
    memref.copy %5, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

