#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_shrink_backward(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?xf32>
    %3 = arith.negf %arg2 : f32
    %v4_pw_single_scalar_0 = arith.constant 0.0 : f32

    %v4_pw_single_pad_3 = arith.constant 0.0 : f32

    %v4_pw_single_pad_4 = arith.constant 0.0 : f32

    %v4_pw_single_pad_5 = arith.constant 0.0 : f32

    %v4_pw_single_pad_6 = arith.constant 0.0 : f32

    %v4_pw_single_pad_7 = arith.constant 0.0 : f32

    %4 = kernel.launch @cudnnPointwiseGraph_f32(%1, %0, %1, %1, %2, %v4_pw_single_scalar_0, %arg2, %3, %v4_pw_single_pad_3, %v4_pw_single_pad_4, %v4_pw_single_pad_5, %v4_pw_single_pad_6, %v4_pw_single_pad_7) {pointwise_graph = array<i64: 216458655187338496, 73198887695551744, 216470749815635968, 76299510486143232, 0, 0, 0, 0, 0, 0, 0, 0>, pointwise_num_nodes = 8 : i64} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, f32, f32, f32, f32, f32, f32, f32, f32) -> tensor<?xf32>
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

