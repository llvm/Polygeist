#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
#map3 = affine_map<(d0, d1, d2) -> (d0 + d1 * 9 + d2 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c42 = arith.constant 42 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.muli %arg2, %c42 : index
      %6 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %7 = arith.index_cast %arg4 : index to i32
        %8 = arith.muli %7, %c6_i32 : i32
        %9 = arith.divsi %8, %c3_i32 : i32
        %10 = arith.addi %7, %c1_i32 : i32
        %11 = arith.muli %10, %c6_i32 : i32
        %12 = arith.addi %11, %c2_i32 : i32
        %13 = arith.divsi %12, %c3_i32 : i32
        %14 = arith.subi %13, %9 : i32
        %15 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %16 = arith.index_cast %arg6 : index to i32
          %17 = arith.muli %16, %c7_i32 : i32
          %18 = arith.divsi %17, %c3_i32 : i32
          %19 = arith.addi %16, %c1_i32 : i32
          %20 = arith.muli %19, %c7_i32 : i32
          %21 = arith.addi %20, %c2_i32 : i32
          %22 = arith.divsi %21, %c3_i32 : i32
          %23 = arith.subi %22, %18 : i32
          %24 = arith.muli %14, %23 : i32
          %25 = arith.sitofp %24 : i32 to f32
          %26 = arith.muli %arg6, %c7 : index
          %27 = arith.cmpi slt, %26, %c0 : index
          %28 = arith.subi %c-1, %26 : index
          %29 = arith.select %27, %28, %26 : index
          %30 = arith.divsi %29, %c3 : index
          %31 = arith.subi %c-1, %30 : index
          %32 = arith.select %27, %31, %30 : index
          %33 = arith.addi %32, %c3 : index
          %34 = affine.for %arg8 = #map1(%arg4) to #map2(%arg4) iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
            %35 = arith.muli %arg8, %c7 : index
            %36 = scf.for %arg10 = %32 to %33 step %c1 iter_args(%arg11 = %arg9) -> (tensor<?xf32>) {
              %37 = affine.apply #map3(%arg6, %arg2, %arg4)
              %extracted = tensor.extract %1[%37] : tensor<?xf32>
              %38 = arith.divf %extracted, %25 : f32
              %39 = arith.addi %arg10, %5 : index
              %40 = arith.addi %39, %35 : index
              %extracted_0 = tensor.extract %arg11[%40] : tensor<?xf32>
              %41 = arith.addf %extracted_0, %38 : f32
              %inserted = tensor.insert %41 into %arg11[%40] : tensor<?xf32>
              scf.yield %inserted : tensor<?xf32>
            }
            affine.yield %36 : tensor<?xf32>
          }
          affine.yield %34 : tensor<?xf32>
        }
        affine.yield %15 : tensor<?xf32>
      }
      affine.yield %6 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

