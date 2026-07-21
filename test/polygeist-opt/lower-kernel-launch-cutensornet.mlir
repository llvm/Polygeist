// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

#psi_a = affine_map<(a, b, c, i, j, k) -> (i + a * 4)>
#psi_b = affine_map<(a, b, c, i, j, k) -> (j + b * 4)>
#psi_c = affine_map<(a, b, c, i, j, k) -> (k + c * 4)>
#u = affine_map<(a, b, c, i, j, k) -> (k + i * 16 + j * 4)>
#out = affine_map<(a, b, c, i, j, k) -> (c + a * 25 + b * 5)>

module {
  kernel.defn @cutensornetTensorProduct3D_f32_tensor(
      %pa: tensor<?x?x?x?x?x?xf32>,
      %pb: tensor<?x?x?x?x?x?xf32>,
      %pc: tensor<?x?x?x?x?x?xf32>,
      %u: tensor<?x?x?x?x?x?xf32>,
      %out: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?x?x?xf32>
  }

  func.func @tensor_product(%psi: tensor<?xf32>, %u: tensor<?xf32>,
                            %out: tensor<?xf32>, %kq: index, %kp: index)
      -> tensor<?xf32> {
    %pa = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_a} : (tensor<?xf32>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf32>
    %pb = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_b} : (tensor<?xf32>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf32>
    %pc = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_c} : (tensor<?xf32>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf32>
    %uv = polygeist.submap(%u, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #u} : (tensor<?xf32>, index, index, index, index, index,
                      index) -> tensor<?x?x?x?x?x?xf32>
    %ov = polygeist.submap(%out, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #out} : (tensor<?xf32>, index, index, index, index, index,
                        index) -> tensor<?x?x?x?x?x?xf32>
    %r = kernel.launch @cutensornetTensorProduct3D_f32_tensor(
        %pa, %pb, %pc, %uv, %ov)
        : (tensor<?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?xf32>,
           tensor<?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?xf32>,
           tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
    %updated = polygeist.submapInverse(
        %out, %r, %kq, %kq, %kq, %kp, %kp, %kp) {map = #out}
        : (tensor<?xf32>, tensor<?x?x?x?x?x?xf32>, index, index, index,
           index, index, index) -> tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @tensor_product
// CHECK: %[[KQ:.*]] = arith.index_cast %arg3 : index to i32
// CHECK: %[[KP:.*]] = arith.index_cast %arg4 : index to i32
// CHECK: call @polygeist_cutensornet_tensor_product_3d_f32(%[[KQ]], %[[KP]],
// CHECK-NOT: kernel.launch
// CHECK-NOT: polygeist.submapInverse
