module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sample_poisson_transform_cpu(%arg0: memref<?xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c31_i32 = arith.constant 31 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %0) -> (tensor<?xi32>) {
      %extracted = tensor.extract %2[%arg3] : tensor<?xf32>
      %5 = arith.negf %extracted : f32
      %6 = math.exp %5 : f32
      %7:3 = scf.while (%arg5 = %c0_i32, %arg6 = %6, %arg7 = %6) : (i32, f32, f32) -> (i32, f32, f32) {
        %8 = arith.cmpi slt, %arg5, %c31_i32 : i32
        %9 = arith.index_cast %arg5 : i32 to index
        %extracted_0 = tensor.extract %1[%arg3, %9] : tensor<?x32xf32>
        %10 = arith.cmpf ogt, %extracted_0, %arg6 : f32
        %11 = arith.addi %arg5, %c1_i32 : i32
        %12 = arith.sitofp %11 : i32 to f32
        %13 = arith.divf %extracted, %12 : f32
        %14 = arith.mulf %arg7, %13 : f32
        %15 = arith.addf %arg6, %14 : f32
        %16 = arith.select %10, %11, %arg5 : i32
        %17 = arith.select %10, %15, %arg6 : f32
        %18 = arith.select %10, %14, %arg7 : f32
        %19 = arith.select %8, %10, %false : i1
        %20 = arith.select %8, %16, %arg5 : i32
        %21 = arith.select %8, %17, %arg6 : f32
        %22 = arith.select %8, %18, %arg7 : f32
        scf.condition(%19) %20, %21, %22 : i32, f32, f32
      } do {
      ^bb0(%arg5: i32, %arg6: f32, %arg7: f32):
        scf.yield %arg5, %arg6, %arg7 : i32, f32, f32
      }
      %inserted = tensor.insert %7#0 into %arg4[%arg3] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

