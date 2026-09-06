#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_trilinear_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x20x24xf32>, %arg2: memref<?x20xf32>, %arg3: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c20 = arith.constant 20 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x20x24xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x20xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x24xf32>
    %extracted_slice = tensor.extract_slice %3[0, 0] [%c8, %c24] [1, 1] : tensor<?x24xf32> to tensor<?x?xf32>
    %aten_network_5_0 = memref.cast %arg0 : memref<?x16xf32> to memref<?x?xf32>

    %aten_network_5_1 = memref.cast %arg1 : memref<?x20x24xf32> to memref<?x?x?xf32>

    %aten_network_5_2 = memref.cast %arg2 : memref<?x20xf32> to memref<?x?xf32>

    %aten_network_5_3 = memref.cast %arg3 : memref<?x24xf32> to memref<?x?xf32>

    kernel.launch @cutensornetNetwork_f32_n3_aten(%aten_network_5_0, %aten_network_5_1, %aten_network_5_2, %aten_network_5_3) {network_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], polygeist.result_destinations = array<i64: 3>} : (memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}

