#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 504 + d1 + d2 * 72 + d3 * 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest_exact3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c14_i32 = arith.constant 14 : i32
    %c3_i32 = arith.constant 3 : i32
    %c16_i32 = arith.constant 16 : i32
    %c18_i32 = arith.constant 18 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.muli %8, %c2_i32 : i32
        %10 = arith.addi %9, %c1_i32 : i32
        %11 = arith.muli %10, %c4_i32 : i32
        %12 = arith.divsi %11, %c14_i32 : i32
        %13 = arith.cmpi sge, %12, %c4_i32 : i32
        %14 = arith.select %13, %c3_i32, %12 : i32
        %15 = arith.addi %6, %14 : i32
        %16 = arith.muli %15, %c5_i32 : i32
        %17 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %18 = arith.index_cast %arg6 : index to i32
          %19 = arith.muli %18, %c2_i32 : i32
          %20 = arith.addi %19, %c1_i32 : i32
          %21 = arith.muli %20, %c5_i32 : i32
          %22 = arith.divsi %21, %c16_i32 : i32
          %23 = arith.cmpi sge, %22, %c5_i32 : i32
          %24 = arith.select %23, %c4_i32, %22 : i32
          %25 = arith.addi %16, %24 : i32
          %26 = arith.muli %25, %c6_i32 : i32
          %27 = affine.for %arg8 = 0 to 9 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
            %28 = arith.index_cast %arg8 : index to i32
            %29 = arith.muli %28, %c2_i32 : i32
            %30 = arith.addi %29, %c1_i32 : i32
            %31 = arith.muli %30, %c6_i32 : i32
            %32 = arith.divsi %31, %c18_i32 : i32
            %33 = arith.cmpi sge, %32, %c6_i32 : i32
            %34 = arith.select %33, %c5_i32, %32 : i32
            %35 = arith.addi %26, %34 : i32
            %36 = arith.index_cast %35 : i32 to index
            %37 = affine.apply #map1(%arg2, %arg8, %arg4, %arg6)
            %extracted = tensor.extract %1[%37] : tensor<?xf32>
            %extracted_0 = tensor.extract %arg9[%36] : tensor<?xf32>
            %38 = arith.addf %extracted_0, %extracted : f32
            %inserted = tensor.insert %38 into %arg9[%36] : tensor<?xf32>
            affine.yield %inserted : tensor<?xf32>
          }
          affine.yield %27 : tensor<?xf32>
        }
        affine.yield %17 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

