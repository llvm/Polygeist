// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

#psi_a = affine_map<(a, b, c, i, j, k) -> (i + a * 4)>
#psi_b = affine_map<(a, b, c, i, j, k) -> (j + b * 4)>
#psi_c = affine_map<(a, b, c, i, j, k) -> (k + c * 4)>
#u = affine_map<(a, b, c, i, j, k) -> (k + i * 16 + j * 4)>
#out = affine_map<(a, b, c, i, j, k) -> (c + a * 25 + b * 5)>
#contract_a = affine_map<(d0, d1, d2, d3, d4) ->
                         (d0 * 60 + d1 * 20 + d3 * 5 + d4)>
#contract_b = affine_map<(d0, d1, d2, d3, d4) ->
                         (d2 * 15 + d3 * 5 + d4)>

module {
  kernel.defn @cutensornetTensorProduct3D_f64_tensor(
      %pa: tensor<?x?x?x?x?x?xf64>,
      %pb: tensor<?x?x?x?x?x?xf64>,
      %pc: tensor<?x?x?x?x?x?xf64>,
      %u: tensor<?x?x?x?x?x?xf64>,
      %out: tensor<?x?x?x?x?x?xf64>) -> tensor<?x?x?x?x?x?xf64> {
    kernel.yield %out : tensor<?x?x?x?x?x?xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r5r5r4(
      %a: tensor<?x?x?x?x?xf64>,
      %b: tensor<?x?x?x?x?xf64>,
      %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?x?xf64>
  }

  func.func @tensor_product_f64(
      %psi: tensor<?xf64>, %u: tensor<?xf64>, %out: tensor<?xf64>,
      %kq: index, %kp: index) -> tensor<?xf64> {
    %pa = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_a} : (tensor<?xf64>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf64>
    %pb = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_b} : (tensor<?xf64>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf64>
    %pc = polygeist.submap(%psi, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #psi_c} : (tensor<?xf64>, index, index, index, index, index,
                          index) -> tensor<?x?x?x?x?x?xf64>
    %uv = polygeist.submap(%u, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #u} : (tensor<?xf64>, index, index, index, index, index,
                      index) -> tensor<?x?x?x?x?x?xf64>
    %ov = polygeist.submap(%out, %kq, %kq, %kq, %kp, %kp, %kp)
        {map = #out} : (tensor<?xf64>, index, index, index, index, index,
                        index) -> tensor<?x?x?x?x?x?xf64>
    %r = kernel.launch @cutensornetTensorProduct3D_f64_tensor(
        %pa, %pb, %pc, %uv, %ov)
        : (tensor<?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?xf64>,
           tensor<?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?xf64>,
           tensor<?x?x?x?x?x?xf64>) -> tensor<?x?x?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %out, %r, %kq, %kq, %kq, %kp, %kp, %kp) {map = #out}
        : (tensor<?xf64>, tensor<?x?x?x?x?x?xf64>, index, index, index,
           index, index, index) -> tensor<?xf64>
    return %updated : tensor<?xf64>
  }

  func.func @contraction_f64(
      %a: tensor<?xf64>, %b: tensor<?xf64>, %c: tensor<?x?x?x?xf64>,
      %n0: index, %n1: index, %n2: index, %n3: index,
      %nk: index) -> tensor<?x?x?x?xf64> {
    %av = polygeist.submap(%a, %n0, %n1, %n2, %n3, %nk)
        {map = #contract_a} : (tensor<?xf64>, index, index, index, index,
                               index) -> tensor<?x?x?x?x?xf64>
    %bv = polygeist.submap(%b, %n0, %n1, %n2, %n3, %nk)
        {map = #contract_b} : (tensor<?xf64>, index, index, index, index,
                               index) -> tensor<?x?x?x?x?xf64>
    %r = kernel.launch @cutensornetContraction2_f64_r5r5r4(%av, %bv, %c)
        {contraction_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]}
        : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>,
           tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    return %r : tensor<?x?x?x?xf64>
  }
}

// CHECK-LABEL: func.func @tensor_product_f64
// CHECK: %[[KQ:.*]] = arith.index_cast %arg3 : index to i32
// CHECK: %[[KP:.*]] = arith.index_cast %arg4 : index to i32
// CHECK: call @polygeist_cutensornet_tensor_product_3d_f64(%[[KQ]], %[[KP]],
// CHECK-NOT: kernel.launch
// CHECK-NOT: polygeist.submapInverse

// CHECK-LABEL: func.func @contraction_f64
// CHECK: memref.alloca() : memref<48xi64>
// CHECK: call @polygeist_cutensornet_contraction2_f64
// CHECK-NOT: kernel.launch
