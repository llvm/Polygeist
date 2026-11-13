#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool3d_legacy_cpu(%arg0: memref<?x8x9x10xf32>, %arg1: memref<?x3x4x5xf32>, %arg2: memref<?x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c-1 = arith.constant -1 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c-11 = arith.constant -11 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x3x4x5xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x4x5xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x8x9x10xf32>
    %3:2 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>) {
      %6:2 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>) {
        %7 = arith.index_cast %arg6 : index to i32
        %8 = arith.muli %7, %c8_i32 : i32
        %9 = arith.divsi %8, %c3_i32 : i32
        %10 = arith.muli %9, %c9_i32 : i32
        %11 = arith.muli %arg6, %c8 : index
        %12 = arith.cmpi slt, %11, %c0 : index
        %13 = arith.subi %c-1, %11 : index
        %14 = arith.select %12, %13, %11 : index
        %15 = arith.divsi %14, %c3 : index
        %16 = arith.subi %c-1, %15 : index
        %17 = arith.select %12, %16, %15 : index
        %18 = arith.addi %11, %c10 : index
        %19 = arith.cmpi slt, %18, %c0 : index
        %20 = arith.subi %c-11, %11 : index
        %21 = arith.select %19, %20, %18 : index
        %22 = arith.divsi %21, %c3 : index
        %23 = arith.subi %c-1, %22 : index
        %24 = arith.select %19, %23, %22 : index
        %25:2 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>) {
          %26 = arith.index_cast %arg9 : index to i32
          %27 = arith.muli %26, %c9_i32 : i32
          %28 = arith.divsi %27, %c4_i32 : i32
          %29 = arith.addi %10, %28 : i32
          %30 = arith.muli %29, %c10_i32 : i32
          %31 = arith.muli %arg9, %c9 : index
          %32 = arith.cmpi slt, %31, %c0 : index
          %33 = arith.subi %c-1, %31 : index
          %34 = arith.select %32, %33, %31 : index
          %35 = arith.divsi %34, %c4 : index
          %36 = arith.subi %c-1, %35 : index
          %37 = arith.select %32, %36, %35 : index
          %38 = arith.addi %37, %c3 : index
          %39:2 = affine.for %arg12 = 0 to 5 iter_args(%arg13 = %arg10, %arg14 = %arg11) -> (tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>) {
            %40 = arith.index_cast %arg12 : index to i32
            %41 = arith.muli %40, %c10_i32 : i32
            %42 = arith.divsi %41, %c5_i32 : i32
            %43 = arith.addi %30, %42 : i32
            %44 = arith.muli %arg12, %c2 : index
            %extracted = tensor.extract %2[%arg3, %17, %37, %44] : tensor<?x8x9x10xf32>
            %45:2 = scf.for %arg15 = %17 to %24 step %c1 iter_args(%arg16 = %extracted, %arg17 = %43) -> (f32, i32) {
              %46 = arith.index_cast %arg15 : index to i32
              %47 = arith.muli %46, %c9_i32 : i32
              %48:2 = scf.for %arg18 = %37 to %38 step %c1 iter_args(%arg19 = %arg16, %arg20 = %arg17) -> (f32, i32) {
                %49 = arith.index_cast %arg18 : index to i32
                %50 = arith.addi %47, %49 : i32
                %51 = arith.muli %50, %c10_i32 : i32
                %alloca = memref.alloca() : memref<f32>
                %52 = bufferization.to_tensor %alloca : memref<f32>
                %inserted_1 = tensor.insert %arg19 into %52[] : tensor<f32>
                %alloca_2 = memref.alloca() : memref<i32>
                %53 = bufferization.to_tensor %alloca_2 : memref<i32>
                %inserted_3 = tensor.insert %arg20 into %53[] : tensor<i32>
                %54:2 = affine.for %arg21 = #map(%arg12) to #map1(%arg12) iter_args(%arg22 = %inserted_1, %arg23 = %inserted_3) -> (tensor<f32>, tensor<i32>) {
                  %extracted_6 = tensor.extract %arg22[] : tensor<f32>
                  %extracted_7 = tensor.extract %arg23[] : tensor<i32>
                  %55 = arith.index_cast %arg21 : index to i32
                  %extracted_8 = tensor.extract %2[%arg3, %arg15, %arg18, %arg21] : tensor<?x8x9x10xf32>
                  %56 = arith.cmpf ogt, %extracted_8, %extracted_6 : f32
                  %57 = arith.select %56, %extracted_8, %extracted_6 : f32
                  %58 = arith.addi %51, %55 : i32
                  %59 = arith.select %56, %58, %extracted_7 : i32
                  %inserted_9 = tensor.insert %57 into %arg22[] : tensor<f32>
                  %inserted_10 = tensor.insert %59 into %arg23[] : tensor<i32>
                  affine.yield %inserted_9, %inserted_10 : tensor<f32>, tensor<i32>
                }
                %extracted_4 = tensor.extract %54#0[] : tensor<f32>
                %extracted_5 = tensor.extract %54#1[] : tensor<i32>
                scf.yield %extracted_4, %extracted_5 : f32, i32
              }
              scf.yield %48#0, %48#1 : f32, i32
            }
            %inserted = tensor.insert %45#0 into %arg13[%arg3, %arg6, %arg9, %arg12] : tensor<?x3x4x5xf32>
            %inserted_0 = tensor.insert %45#1 into %arg14[%arg3, %arg6, %arg9, %arg12] : tensor<?x3x4x5xi32>
            affine.yield %inserted, %inserted_0 : tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>
          }
          affine.yield %39#0, %39#1 : tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>
        }
        affine.yield %25#0, %25#1 : tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>
      }
      affine.yield %6#0, %6#1 : tensor<?x3x4x5xf32>, tensor<?x3x4x5xi32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?x3x4x5xi32>
    memref.copy %4, %arg2 : memref<?x3x4x5xi32> to memref<?x3x4x5xi32>
    %5 = bufferization.to_memref %3#0 : memref<?x3x4x5xf32>
    memref.copy %5, %arg1 : memref<?x3x4x5xf32> to memref<?x3x4x5xf32>
    return
  }
}

