#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 27 + d1 + d2 * 9 + d3 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c336 = arith.constant 336 : index
    %c56 = arith.constant 56 : index
    %c-11 = arith.constant -11 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %4 = arith.muli %arg2, %c336 : index
      %5 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %6 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %7 = arith.muli %arg6, %c7 : index
          %8 = arith.cmpi slt, %7, %c0 : index
          %9 = arith.subi %c-1, %7 : index
          %10 = arith.select %8, %9, %7 : index
          %11 = arith.divsi %10, %c3 : index
          %12 = arith.subi %c-1, %11 : index
          %13 = arith.select %8, %12, %11 : index
          %14 = arith.addi %13, %c3 : index
          %alloca = memref.alloca(%c3) : memref<?xi32>
          %15 = bufferization.to_tensor %alloca : memref<?xi32>
          %alloca_0 = memref.alloca(%c3) : memref<?xf32>
          %16 = bufferization.to_tensor %alloca_0 : memref<?xf32>
          %17:3 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %15, %arg10 = %16, %arg11 = %arg7) -> (tensor<?xi32>, tensor<?xf32>, tensor<?xf32>) {
            %18 = arith.index_cast %arg8 : index to i32
            %19 = arith.muli %18, %c8_i32 : i32
            %20 = arith.divsi %19, %c3_i32 : i32
            %21 = arith.addi %18, %c1_i32 : i32
            %22 = arith.muli %21, %c8_i32 : i32
            %23 = arith.addi %22, %c2_i32 : i32
            %24 = arith.divsi %23, %c3_i32 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %20 : i32 to index
            %27 = arith.subi %25, %26 : index
            %28 = arith.muli %arg8, %c8 : index
            %29 = arith.cmpi slt, %28, %c0 : index
            %30 = arith.subi %c-1, %28 : index
            %31 = arith.select %29, %30, %28 : index
            %32 = arith.divsi %31, %c3 : index
            %33 = arith.subi %c-1, %32 : index
            %34 = arith.select %29, %33, %32 : index
            %35 = arith.addi %28, %c10 : index
            %36 = arith.cmpi slt, %35, %c0 : index
            %37 = arith.subi %c-11, %28 : index
            %38 = arith.select %36, %37, %35 : index
            %39 = arith.divsi %38, %c3 : index
            %40 = arith.subi %c-1, %39 : index
            %41 = arith.select %36, %40, %39 : index
            %inserted = tensor.insert %c0_i32 into %arg9[%arg8] : tensor<?xi32>
            %inserted_1 = tensor.insert %cst into %arg10[%arg8] : tensor<?xf32>
            %42:2 = affine.for %arg12 = #map(%arg4) to #map1(%arg4) iter_args(%arg13 = %inserted, %arg14 = %inserted_1) -> (tensor<?xi32>, tensor<?xf32>) {
              %extracted_4 = tensor.extract %arg13[%arg8] : tensor<?xi32>
              %extracted_5 = tensor.extract %arg14[%arg8] : tensor<?xf32>
              %46 = arith.muli %arg12, %c56 : index
              %47:2 = scf.for %arg15 = %13 to %14 step %c1 iter_args(%arg16 = %extracted_4, %arg17 = %extracted_5) -> (i32, f32) {
                %48 = arith.index_cast %arg16 : i32 to index
                %49 = arith.addi %48, %27 : index
                %50 = arith.index_cast %49 : index to i32
                %51 = arith.muli %arg15, %c8 : index
                %52 = scf.for %arg18 = %34 to %41 step %c1 iter_args(%arg19 = %arg17) -> (f32) {
                  %53 = arith.addi %arg18, %51 : index
                  %54 = arith.addi %53, %4 : index
                  %55 = arith.addi %54, %46 : index
                  %extracted_8 = tensor.extract %1[%55] : tensor<?xf32>
                  %56 = arith.addf %arg19, %extracted_8 : f32
                  scf.yield %56 : f32
                }
                scf.yield %50, %52 : i32, f32
              }
              %inserted_6 = tensor.insert %47#0 into %arg13[%arg8] : tensor<?xi32>
              %inserted_7 = tensor.insert %47#1 into %arg14[%arg8] : tensor<?xf32>
              affine.yield %inserted_6, %inserted_7 : tensor<?xi32>, tensor<?xf32>
            }
            %extracted = tensor.extract %42#0[%arg8] : tensor<?xi32>
            %extracted_2 = tensor.extract %42#1[%arg8] : tensor<?xf32>
            %43 = arith.sitofp %extracted : i32 to f32
            %44 = arith.divf %extracted_2, %43 : f32
            %45 = affine.apply #map2(%arg2, %arg8, %arg4, %arg6)
            %inserted_3 = tensor.insert %44 into %arg11[%45] : tensor<?xf32>
            affine.yield %42#0, %42#1, %inserted_3 : tensor<?xi32>, tensor<?xf32>, tensor<?xf32>
          }
          affine.yield %17#2 : tensor<?xf32>
        }
        affine.yield %6 : tensor<?xf32>
      }
      affine.yield %5 : tensor<?xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?xf32>
    memref.copy %3, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

