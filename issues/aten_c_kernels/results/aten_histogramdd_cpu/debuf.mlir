#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_histogramdd_cpu(%arg0: memref<?x2xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: memref<?x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c12_i32 = arith.constant 12 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.600000e+01 : f32
    %cst_1 = arith.constant 1.200000e+01 : f32
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x12xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x2xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c16, %c12] [1, 1] : tensor<?x12xf32> to tensor<?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %0[0, 0] [%c16, %c12] [1, 1] : tensor<?x?xf32> into tensor<?x12xf32>
    %4 = arith.subf %arg3, %arg2 : f32
    %5 = arith.subf %arg5, %arg4 : f32
    %6 = affine.for %arg7 = 0 to 4096 iter_args(%arg8 = %inserted_slice) -> (tensor<?x12xf32>) {
      %extracted = tensor.extract %2[%arg7, %c0] : tensor<?x2xf32>
      %8 = arith.subf %extracted, %arg2 : f32
      %9 = arith.mulf %8, %cst_0 : f32
      %10 = arith.divf %9, %4 : f32
      %11 = arith.fptosi %10 : f32 to i32
      %extracted_2 = tensor.extract %2[%arg7, %c1] : tensor<?x2xf32>
      %12 = arith.subf %extracted_2, %arg4 : f32
      %13 = arith.mulf %12, %cst_1 : f32
      %14 = arith.divf %13, %5 : f32
      %15 = arith.fptosi %14 : f32 to i32
      %16 = arith.cmpi sge, %11, %c0_i32 : i32
      %17 = arith.cmpi slt, %11, %c16_i32 : i32
      %18 = arith.cmpi sge, %15, %c0_i32 : i32
      %19 = arith.cmpi slt, %15, %c12_i32 : i32
      %20 = arith.andi %18, %19 : i1
      %21 = arith.andi %17, %20 : i1
      %22 = arith.andi %16, %21 : i1
      %23 = scf.if %22 -> (tensor<?x12xf32>) {
        %24 = arith.index_cast %11 : i32 to index
        %25 = arith.index_cast %15 : i32 to index
        %extracted_3 = tensor.extract %1[%arg7] : tensor<?xf32>
        %extracted_4 = tensor.extract %arg8[%24, %25] : tensor<?x12xf32>
        %26 = arith.addf %extracted_4, %extracted_3 : f32
        %inserted = tensor.insert %26 into %arg8[%24, %25] : tensor<?x12xf32>
        scf.yield %inserted : tensor<?x12xf32>
      } else {
        scf.yield %arg8 : tensor<?x12xf32>
      }
      affine.yield %23 : tensor<?x12xf32>
    }
    %7 = bufferization.to_memref %6 : memref<?x12xf32>
    memref.copy %7, %arg6 : memref<?x12xf32> to memref<?x12xf32>
    return
  }
}

