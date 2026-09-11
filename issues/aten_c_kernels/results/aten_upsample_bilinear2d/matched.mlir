#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d(%arg0: memref<?x3x4x4xf32>, %arg1: memref<?x3x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x3x8x8xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [%c2, %c3, %c8, %c8] [1, 1, 1, 1] : tensor<?x3x8x8xf32> to tensor<?x?x?x?xf32>
    %bilinear2x_1873_0 = arith.constant 6 : i32
    %bilinear2x_1873_1 = arith.constant 2 : i32
    %bilinear2x_1873_2 = arith.constant 6 : i32
    %bilinear2x_1873_3 = arith.constant 1 : i32
    %bilinear2x_1873_4 = arith.constant 4 : i32
    %bilinear2x_1873_5 = arith.constant 4 : i32
    %bilinear2x_1873_6 = arith.constant 1 : i32
    %bilinear2x_1873_7 = arith.constant 8 : i32
    %bilinear2x_1873_8 = arith.constant 8 : i32
    %bilinear2x_1873_9 = arith.constant 1 : i32
    kernel.launch @cudnnBilinearUpsample2x_f32_r4(%bilinear2x_1873_0, %bilinear2x_1873_1, %bilinear2x_1873_2, %bilinear2x_1873_3, %bilinear2x_1873_4, %bilinear2x_1873_5, %bilinear2x_1873_6, %bilinear2x_1873_7, %bilinear2x_1873_8, %bilinear2x_1873_9, %arg0, %arg1) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<?x3x4x4xf32>, memref<?x3x8x8xf32>) -> ()
    return
  }
}

