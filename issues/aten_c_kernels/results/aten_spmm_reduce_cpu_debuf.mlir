#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: i32, %arg5: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %0 = bufferization.to_tensor %arg5 : memref<?x24xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?x24xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?xi32>
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = arith.cmpi eq, %arg4, %c2_i32 : i32
    %7 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %8 = arith.cmpi eq, %arg4, %c3_i32 : i32
    %9 = arith.select %8, %cst_0, %cst : f32
    %10 = arith.select %6, %cst_1, %9 : f32
    %11 = arith.cmpi eq, %arg4, %c1_i32 : i32
    %12 = arith.select %7, %true, %11 : i1
    %13 = arith.addi %5, %c-1 : index
    %14 = arith.cmpi eq, %13, %c0 : index
    %15 = affine.for %arg6 = 0 to 16 iter_args(%arg7 = %0) -> (tensor<?x24xf32>) {
      %17 = affine.for %arg8 = 0 to 24 iter_args(%arg9 = %arg7) -> (tensor<?x24xf32>) {
        %extracted = tensor.extract %4[%arg6] : tensor<?xi32>
        %18 = affine.apply #map(%arg6)
        %extracted_2 = tensor.extract %4[%18] : tensor<?xi32>
        %19 = arith.index_cast %extracted_2 : i32 to index
        %20 = arith.index_cast %extracted : i32 to index
        %21 = scf.for %arg10 = %20 to %19 step %c1 iter_args(%arg11 = %10) -> (f32) {
          %extracted_3 = tensor.extract %2[%arg10] : tensor<?xf32>
          %extracted_4 = tensor.extract %3[%arg10] : tensor<?xi32>
          %28 = arith.index_cast %extracted_4 : i32 to index
          %extracted_5 = tensor.extract %1[%28, %arg8] : tensor<?x24xf32>
          %29 = arith.mulf %extracted_3, %extracted_5 : f32
          %30 = scf.if %12 -> (f32) {
            %31 = arith.addf %arg11, %29 : f32
            scf.yield %31 : f32
          } else {
            %31 = affine.apply #map1()[%5]
            %32 = arith.cmpi eq, %31, %c0 : index
            %33 = arith.cmpf ogt, %arg11, %29 : f32
            %34 = arith.select %33, %arg11, %29 : f32
            %35 = arith.cmpf olt, %arg11, %29 : f32
            %36 = arith.select %35, %arg11, %29 : f32
            %37 = arith.select %32, %34, %36 : f32
            scf.yield %37 : f32
          }
          scf.yield %30 : f32
        }
        %22 = arith.cmpi sgt, %extracted_2, %extracted : i32
        %23 = arith.andi %14, %22 : i1
        %24 = arith.subi %extracted_2, %extracted : i32
        %25 = arith.sitofp %24 : i32 to f32
        %26 = arith.divf %21, %25 : f32
        %27 = arith.select %23, %26, %21 : f32
        %inserted = tensor.insert %27 into %arg9[%arg6, %arg8] : tensor<?x24xf32>
        affine.yield %inserted : tensor<?x24xf32>
      }
      affine.yield %17 : tensor<?x24xf32>
    }
    %16 = bufferization.to_memref %15 : memref<?x24xf32>
    memref.copy %16, %arg5 : memref<?x24xf32> to memref<?x24xf32>
    return
  }
}

