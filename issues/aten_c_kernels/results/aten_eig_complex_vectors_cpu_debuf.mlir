#map = affine_map<(d0) -> (-d0 + 62)>
#map1 = affine_map<(d0, d1) -> (d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_eig_complex_vectors_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>, %arg2: memref<?x64xf32>, %arg3: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %4:2 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x64xf32>, tensor<?x64xf32>) {
      %7:2 = affine.for %arg7 = 0 to 64 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<?x64xf32>, tensor<?x64xf32>) {
        %extracted = tensor.extract %3[%arg4, %arg7] : tensor<?x64xf32>
        %inserted = tensor.insert %extracted into %arg8[%arg4, %arg7] : tensor<?x64xf32>
        %extracted_0 = tensor.extract %2[%arg7] : tensor<?xf32>
        %8 = arith.cmpf oeq, %extracted_0, %cst : f32
        %9 = scf.if %8 -> (f32) {
          scf.yield %cst : f32
        } else {
          %10 = affine.apply #map(%arg7)
          %11 = arith.cmpi sge, %10, %c0 : index
          %12 = affine.apply #map1(%arg4, %arg7)
          %extracted_2 = tensor.extract %3[%arg4, %12] : tensor<?x64xf32>
          %13 = arith.select %11, %extracted_2, %cst : f32
          scf.yield %13 : f32
        }
        %inserted_1 = tensor.insert %9 into %arg9[%arg4, %arg7] : tensor<?x64xf32>
        affine.yield %inserted, %inserted_1 : tensor<?x64xf32>, tensor<?x64xf32>
      }
      affine.yield %7#0, %7#1 : tensor<?x64xf32>, tensor<?x64xf32>
    }
    %5 = bufferization.to_memref %4#1 : memref<?x64xf32>
    memref.copy %5, %arg3 : memref<?x64xf32> to memref<?x64xf32>
    %6 = bufferization.to_memref %4#0 : memref<?x64xf32>
    memref.copy %6, %arg2 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

