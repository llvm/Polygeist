#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0 * 27 + d1 + d2 * 9 + d3 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c336 = arith.constant 336 : index
    %c56 = arith.constant 56 : index
    %c-11 = arith.constant -11 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.muli %arg2, %c336 : index
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
          %25 = arith.muli %arg6, %c7 : index
          %26 = arith.cmpi slt, %25, %c0 : index
          %27 = arith.subi %c-1, %25 : index
          %28 = arith.select %26, %27, %25 : index
          %29 = arith.divsi %28, %c3 : index
          %30 = arith.subi %c-1, %29 : index
          %31 = arith.select %26, %30, %29 : index
          %32 = arith.addi %31, %c3 : index
          %33 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
            %34 = arith.index_cast %arg8 : index to i32
            %35 = arith.muli %34, %c8_i32 : i32
            %36 = arith.divsi %35, %c3_i32 : i32
            %37 = arith.addi %34, %c1_i32 : i32
            %38 = arith.muli %37, %c8_i32 : i32
            %39 = arith.addi %38, %c2_i32 : i32
            %40 = arith.divsi %39, %c3_i32 : i32
            %41 = arith.subi %40, %36 : i32
            %42 = arith.muli %24, %41 : i32
            %43 = arith.sitofp %42 : i32 to f32
            %44 = arith.muli %arg8, %c8 : index
            %45 = arith.cmpi slt, %44, %c0 : index
            %46 = arith.subi %c-1, %44 : index
            %47 = arith.select %45, %46, %44 : index
            %48 = arith.divsi %47, %c3 : index
            %49 = arith.subi %c-1, %48 : index
            %50 = arith.select %45, %49, %48 : index
            %51 = arith.addi %44, %c10 : index
            %52 = arith.cmpi slt, %51, %c0 : index
            %53 = arith.subi %c-11, %44 : index
            %54 = arith.select %52, %53, %51 : index
            %55 = arith.divsi %54, %c3 : index
            %56 = arith.subi %c-1, %55 : index
            %57 = arith.select %52, %56, %55 : index
            %58 = affine.for %arg10 = #map1(%arg4) to #map2(%arg4) iter_args(%arg11 = %arg9) -> (tensor<?xf32>) {
              %59 = arith.muli %arg10, %c56 : index
              %60 = scf.for %arg12 = %31 to %32 step %c1 iter_args(%arg13 = %arg11) -> (tensor<?xf32>) {
                %61 = arith.muli %arg12, %c8 : index
                %62 = scf.for %arg14 = %50 to %57 step %c1 iter_args(%arg15 = %arg13) -> (tensor<?xf32>) {
                  %63 = affine.apply #map3(%arg2, %arg8, %arg4, %arg6)
                  %extracted = tensor.extract %1[%63] : tensor<?xf32>
                  %64 = arith.divf %extracted, %43 : f32
                  %65 = arith.addi %arg14, %61 : index
                  %66 = arith.addi %65, %5 : index
                  %67 = arith.addi %66, %59 : index
                  %extracted_0 = tensor.extract %arg15[%67] : tensor<?xf32>
                  %68 = arith.addf %extracted_0, %64 : f32
                  %inserted = tensor.insert %68 into %arg15[%67] : tensor<?xf32>
                  scf.yield %inserted : tensor<?xf32>
                }
                scf.yield %62 : tensor<?xf32>
              }
              affine.yield %60 : tensor<?xf32>
            }
            affine.yield %58 : tensor<?xf32>
          }
          affine.yield %33 : tensor<?xf32>
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

