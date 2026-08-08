#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 27 + d1 + d2 * 9 + d3 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c336 = arith.constant 336 : index
    %c56 = arith.constant 56 : index
    %c-11 = arith.constant -11 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3:2 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?xf32>, tensor<?xi32>) {
      %6 = arith.muli %arg3, %c336 : index
      %7:2 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?xf32>, tensor<?xi32>) {
        %8:2 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?xf32>, tensor<?xi32>) {
          %9 = arith.muli %arg9, %c7 : index
          %10 = arith.cmpi slt, %9, %c0 : index
          %11 = arith.subi %c-1, %9 : index
          %12 = arith.select %10, %11, %9 : index
          %13 = arith.divsi %12, %c3 : index
          %14 = arith.subi %c-1, %13 : index
          %15 = arith.select %10, %14, %13 : index
          %16 = arith.addi %15, %c3 : index
          %alloca = memref.alloca(%c3) : memref<?xi32>
          %17 = bufferization.to_tensor %alloca : memref<?xi32>
          %alloca_0 = memref.alloca(%c3) : memref<?xf32>
          %18 = bufferization.to_tensor %alloca_0 : memref<?xf32>
          %19:4 = affine.for %arg12 = 0 to 3 iter_args(%arg13 = %17, %arg14 = %18, %arg15 = %arg10, %arg16 = %arg11) -> (tensor<?xi32>, tensor<?xf32>, tensor<?xf32>, tensor<?xi32>) {
            %20 = arith.muli %arg12, %c8 : index
            %21 = arith.cmpi slt, %20, %c0 : index
            %22 = arith.subi %c-1, %20 : index
            %23 = arith.select %21, %22, %20 : index
            %24 = arith.divsi %23, %c3 : index
            %25 = arith.subi %c-1, %24 : index
            %26 = arith.select %21, %25, %24 : index
            %27 = arith.addi %20, %c10 : index
            %28 = arith.cmpi slt, %27, %c0 : index
            %29 = arith.subi %c-11, %20 : index
            %30 = arith.select %28, %29, %27 : index
            %31 = arith.divsi %30, %c3 : index
            %32 = arith.subi %c-1, %31 : index
            %33 = arith.select %28, %32, %31 : index
            %inserted = tensor.insert %c0_i32 into %arg13[%arg12] : tensor<?xi32>
            %inserted_1 = tensor.insert %cst into %arg14[%arg12] : tensor<?xf32>
            %34:2 = affine.for %arg17 = #map(%arg6) to #map1(%arg6) iter_args(%arg18 = %inserted, %arg19 = %inserted_1) -> (tensor<?xi32>, tensor<?xf32>) {
              %extracted_5 = tensor.extract %arg18[%arg12] : tensor<?xi32>
              %extracted_6 = tensor.extract %arg19[%arg12] : tensor<?xf32>
              %37 = arith.index_cast %arg17 : index to i32
              %38 = arith.muli %37, %c7_i32 : i32
              %39 = arith.muli %arg17, %c56 : index
              %40:2 = scf.for %arg20 = %15 to %16 step %c1 iter_args(%arg21 = %extracted_5, %arg22 = %extracted_6) -> (i32, f32) {
                %41 = arith.index_cast %arg20 : index to i32
                %42 = arith.addi %38, %41 : i32
                %43 = arith.muli %42, %c8_i32 : i32
                %44 = arith.muli %arg20, %c8 : index
                %45:2 = scf.for %arg23 = %26 to %33 step %c1 iter_args(%arg24 = %arg21, %arg25 = %arg22) -> (i32, f32) {
                  %46 = arith.index_cast %arg23 : index to i32
                  %47 = arith.addi %arg23, %44 : index
                  %48 = arith.addi %47, %6 : index
                  %49 = arith.addi %48, %39 : index
                  %extracted_9 = tensor.extract %2[%49] : tensor<?xf32>
                  %50 = arith.cmpf ogt, %extracted_9, %arg25 : f32
                  %51 = arith.select %50, %extracted_9, %arg25 : f32
                  %52 = arith.addi %43, %46 : i32
                  %53 = arith.select %50, %52, %arg24 : i32
                  scf.yield %53, %51 : i32, f32
                }
                scf.yield %45#0, %45#1 : i32, f32
              }
              %inserted_7 = tensor.insert %40#0 into %arg18[%arg12] : tensor<?xi32>
              %inserted_8 = tensor.insert %40#1 into %arg19[%arg12] : tensor<?xf32>
              affine.yield %inserted_7, %inserted_8 : tensor<?xi32>, tensor<?xf32>
            }
            %extracted = tensor.extract %34#0[%arg12] : tensor<?xi32>
            %extracted_2 = tensor.extract %34#1[%arg12] : tensor<?xf32>
            %35 = affine.apply #map2(%arg3, %arg12, %arg6, %arg9)
            %inserted_3 = tensor.insert %extracted_2 into %arg15[%35] : tensor<?xf32>
            %36 = affine.apply #map2(%arg3, %arg12, %arg6, %arg9)
            %inserted_4 = tensor.insert %extracted into %arg16[%36] : tensor<?xi32>
            affine.yield %34#0, %34#1, %inserted_3, %inserted_4 : tensor<?xi32>, tensor<?xf32>, tensor<?xf32>, tensor<?xi32>
          }
          affine.yield %19#2, %19#3 : tensor<?xf32>, tensor<?xi32>
        }
        affine.yield %8#0, %8#1 : tensor<?xf32>, tensor<?xi32>
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

