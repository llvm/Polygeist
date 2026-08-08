module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_amp_update_scale_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %extracted = tensor.extract %0[%c0] : tensor<?xf32>
    %3 = arith.cmpf une, %extracted, %cst : f32
    %4:2 = scf.if %3 -> (tensor<?xf32>, tensor<?xi32>) {
      %extracted_0 = tensor.extract %2[%c0] : tensor<?xf32>
      %7 = arith.mulf %extracted_0, %arg4 : f32
      %inserted = tensor.insert %7 into %2[%c0] : tensor<?xf32>
      %inserted_1 = tensor.insert %c0_i32 into %1[%c0] : tensor<?xi32>
      scf.yield %inserted, %inserted_1 : tensor<?xf32>, tensor<?xi32>
    } else {
      %extracted_0 = tensor.extract %1[%c0] : tensor<?xi32>
      %7 = arith.addi %extracted_0, %c1_i32 : i32
      %8 = arith.cmpi eq, %7, %arg5 : i32
      %9:2 = scf.if %8 -> (tensor<?xf32>, tensor<?xi32>) {
        %extracted_1 = tensor.extract %2[%c0] : tensor<?xf32>
        %10 = arith.mulf %extracted_1, %arg3 : f32
        %inserted = tensor.insert %10 into %2[%c0] : tensor<?xf32>
        %inserted_2 = tensor.insert %c0_i32 into %1[%c0] : tensor<?xi32>
        scf.yield %inserted, %inserted_2 : tensor<?xf32>, tensor<?xi32>
      } else {
        %inserted = tensor.insert %7 into %1[%c0] : tensor<?xi32>
        scf.yield %2, %inserted : tensor<?xf32>, tensor<?xi32>
      }
      scf.yield %9#0, %9#1 : tensor<?xf32>, tensor<?xi32>
    }
    %5 = bufferization.to_memref %4#1 : memref<?xi32>
    memref.copy %5, %arg1 : memref<?xi32> to memref<?xi32>
    %6 = bufferization.to_memref %4#0 : memref<?xf32>
    memref.copy %6, %arg0 : memref<?xf32> to memref<?xf32>
    return
  }
}

