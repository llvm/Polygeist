#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 720 + d1 + d2 * 90 + d3 * 10)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c3_i32 = arith.constant 3 : i32
    %false = arith.constant false
    %c-2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.addi %8, %c-2_i32 : i32
        %10 = arith.cmpi slt, %9, %c0_i32 : i32
        %11 = arith.select %10, %c0_i32, %9 : i32
        %12 = affine.apply #map1(%arg4)
        %13 = arith.cmpi sge, %12, %c0 : index
        %14 = arith.cmpi sge, %9, %c4_i32 : i32
        %15 = arith.select %13, %false, %14 : i1
        %16 = arith.select %15, %c3_i32, %11 : i32
        %17 = arith.addi %6, %16 : i32
        %18 = arith.muli %17, %c5_i32 : i32
        %19 = affine.for %arg6 = 0 to 9 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %20 = arith.index_cast %arg6 : index to i32
          %21 = arith.addi %20, %c-2_i32 : i32
          %22 = arith.cmpi slt, %21, %c0_i32 : i32
          %23 = arith.select %22, %c0_i32, %21 : i32
          %24 = affine.apply #map1(%arg6)
          %25 = arith.cmpi sge, %24, %c0 : index
          %26 = arith.cmpi sge, %21, %c5_i32 : i32
          %27 = arith.select %25, %false, %26 : i1
          %28 = arith.select %27, %c4_i32, %23 : i32
          %29 = arith.addi %18, %28 : i32
          %30 = arith.muli %29, %c6_i32 : i32
          %31 = affine.for %arg8 = 0 to 10 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
            %32 = arith.index_cast %arg8 : index to i32
            %33 = arith.addi %32, %c-2_i32 : i32
            %34 = arith.cmpi slt, %33, %c0_i32 : i32
            %35 = arith.select %34, %c0_i32, %33 : i32
            %36 = affine.apply #map1(%arg8)
            %37 = arith.cmpi sge, %36, %c0 : index
            %38 = arith.cmpi sge, %33, %c6_i32 : i32
            %39 = arith.select %37, %false, %38 : i1
            %40 = arith.select %39, %c5_i32, %35 : i32
            %41 = arith.addi %30, %40 : i32
            %42 = arith.index_cast %41 : i32 to index
            %43 = affine.apply #map2(%arg2, %arg8, %arg4, %arg6)
            %extracted = tensor.extract %1[%43] : tensor<?xf32>
            %extracted_0 = tensor.extract %arg9[%42] : tensor<?xf32>
            %44 = arith.addf %extracted_0, %extracted : f32
            %inserted = tensor.insert %44 into %arg9[%42] : tensor<?xf32>
            affine.yield %inserted : tensor<?xf32>
          }
          affine.yield %31 : tensor<?xf32>
        }
        affine.yield %19 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

