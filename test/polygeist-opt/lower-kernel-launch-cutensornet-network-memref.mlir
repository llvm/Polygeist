// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

#x = affine_map<(b, o, i, j) -> (b, i)>
#w = affine_map<(b, o, i, j) -> (o, i, j)>
#y = affine_map<(b, o, i, j) -> (b, j)>
#out = affine_map<(b, o, i, j) -> (b, o)>

module {
  kernel.defn @cutensornetNetwork_f32_n3_aten(
      %x: memref<?x?xf32>, %w: memref<?x?x?xf32>,
      %y: memref<?x?xf32>, %out: memref<?x?xf32>) { kernel.yield }

  func.func @three_input_network(
      %x: memref<?x?xf32>, %w: memref<?x?x?xf32>,
      %y: memref<?x?xf32>, %out: memref<?x?xf32>) {
    kernel.launch @cutensornetNetwork_f32_n3_aten(%x, %w, %y, %out)
        {network_maps = [#x, #w, #y, #out],
         polygeist.result_destinations = array<i64: 3>} :
        (memref<?x?xf32>, memref<?x?x?xf32>,
         memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @three_input_network
// CHECK: memref.alloca() : memref<34xi64>
// CHECK: memref.alloca() : memref<4xi64>
// CHECK: call @polygeist_cutensornet_network_f32
// CHECK-NOT: kernel.launch
