#a_flat = affine_map<(d0, d1, d2, d3) ->
                     (d0 * 24 + d1 * 8 + d2 * 2 + d3)>
#b_flat = affine_map<(d0, d1, d2, d3, d4) ->
                     (d0 * 48 + d1 * 24 + d2 * 12 + d3 * 4 + d4)>
#c_flat = affine_map<(d0, d1, d2, d3) ->
                     (d0 * 12 + d1 * 4 + d2 * 2 + d3)>
#a5_flat = affine_map<(d0, d1, d2, d3, d4) ->
                      (d0 * 96 + d1 * 32 + d2 * 16 + d3 * 8 + d4)>
#b4_flat = affine_map<(d0, d1, d2, d3) ->
                      (d0 * 24 + d1 * 8 + d2 * 2 + d3)>
#a5_broadcast_flat = affine_map<(d0, d1, d2, d3, d4) ->
                                (d0 * 24 + d1 * 8 + d3 * 4 + d4)>
#b5_broadcast_flat = affine_map<(d0, d1, d2, d3, d4) ->
                                (d2 * 8 + d3 * 4 + d4)>

module {
  kernel.defn @cutensornetContraction2_f64_r4r5r4(
      %a: tensor<?x?x?x?xf64>,
      %b: tensor<?x?x?x?x?xf64>,
      %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?x?xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r5r4r4(
      %a: tensor<?x?x?x?x?xf64>,
      %b: tensor<?x?x?x?xf64>,
      %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?x?xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r5r5r4(
      %a: tensor<?x?x?x?x?xf64>,
      %b: tensor<?x?x?x?x?xf64>,
      %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?x?xf64>
  }

  func.func @mfem_cutensornet_r4r5r4(
      %a_memref: memref<?xf64>, %b_memref: memref<?xf64>,
      %c_memref: memref<?xf64>) {
    %n0 = arith.constant 2 : index
    %n1 = arith.constant 3 : index
    %n2 = arith.constant 2 : index
    %n3 = arith.constant 2 : index
    %nk = arith.constant 4 : index
    %a = bufferization.to_tensor %a_memref : memref<?xf64>
    %b = bufferization.to_tensor %b_memref : memref<?xf64>
    %c = bufferization.to_tensor %c_memref : memref<?xf64>
    %av = polygeist.submap(%a, %n0, %n1, %nk, %n2) {map = #a_flat}
        : (tensor<?xf64>, index, index, index, index)
          -> tensor<?x?x?x?xf64>
    %bv = polygeist.submap(%b, %n0, %n3, %n2, %n1, %nk) {map = #b_flat}
        : (tensor<?xf64>, index, index, index, index, index)
          -> tensor<?x?x?x?x?xf64>
    %cv = polygeist.submap(%c, %n0, %n1, %n3, %n2) {map = #c_flat}
        : (tensor<?xf64>, index, index, index, index)
          -> tensor<?x?x?x?xf64>
    %result = kernel.launch @cutensornetContraction2_f64_r4r5r4(
        %av, %bv, %cv) {contraction_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]}
        : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>,
           tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %c, %result, %n0, %n1, %n3, %n2) {map = #c_flat}
        : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index)
          -> tensor<?xf64>
    %updated_memref = bufferization.to_memref %updated : memref<?xf64>
    memref.copy %updated_memref, %c_memref
        : memref<?xf64> to memref<?xf64>
    return
  }

  func.func @mfem_cutensornet_r5r4r4(
      %a_memref: memref<?xf64>, %b_memref: memref<?xf64>,
      %c_memref: memref<?xf64>) {
    %n0 = arith.constant 2 : index
    %n1 = arith.constant 3 : index
    %n2 = arith.constant 2 : index
    %n3 = arith.constant 2 : index
    %nk = arith.constant 4 : index
    %a = bufferization.to_tensor %a_memref : memref<?xf64>
    %b = bufferization.to_tensor %b_memref : memref<?xf64>
    %c = bufferization.to_tensor %c_memref : memref<?xf64>
    %av = polygeist.submap(%a, %n0, %n1, %n2, %n3, %nk) {map = #a5_flat}
        : (tensor<?xf64>, index, index, index, index, index)
          -> tensor<?x?x?x?x?xf64>
    %bv = polygeist.submap(%b, %n0, %n1, %nk, %n3) {map = #b4_flat}
        : (tensor<?xf64>, index, index, index, index)
          -> tensor<?x?x?x?xf64>
    %cv = polygeist.submap(%c, %n0, %n1, %n2, %n3) {map = #c_flat}
        : (tensor<?xf64>, index, index, index, index)
          -> tensor<?x?x?x?xf64>
    %result = kernel.launch @cutensornetContraction2_f64_r5r4r4(
        %av, %bv, %cv) {contraction_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]}
        : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>,
           tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %c, %result, %n0, %n1, %n2, %n3) {map = #c_flat}
        : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index)
          -> tensor<?xf64>
    %updated_memref = bufferization.to_memref %updated : memref<?xf64>
    memref.copy %updated_memref, %c_memref
        : memref<?xf64> to memref<?xf64>
    return
  }

  func.func @mfem_cutensornet_r5r5r4_broadcast(
      %a_memref: memref<?xf64>, %b_memref: memref<?xf64>,
      %c_memref: memref<?xf64>) {
    %n0 = arith.constant 2 : index
    %n1 = arith.constant 3 : index
    %n2 = arith.constant 2 : index
    %n3 = arith.constant 2 : index
    %nk = arith.constant 4 : index
    %a = bufferization.to_tensor %a_memref : memref<?xf64>
    %b = bufferization.to_tensor %b_memref : memref<?xf64>
    %c = bufferization.to_tensor %c_memref : memref<?xf64>
    %av = polygeist.submap(%a, %n0, %n1, %n2, %n3, %nk)
        {map = #a5_broadcast_flat}
        : (tensor<?xf64>, index, index, index, index, index)
          -> tensor<?x?x?x?x?xf64>
    %bv = polygeist.submap(%b, %n0, %n1, %n2, %n3, %nk)
        {map = #b5_broadcast_flat}
        : (tensor<?xf64>, index, index, index, index, index)
          -> tensor<?x?x?x?x?xf64>
    %cv = polygeist.submap(%c, %n0, %n1, %n2, %n3) {map = #c_flat}
        : (tensor<?xf64>, index, index, index, index)
          -> tensor<?x?x?x?xf64>
    %result = kernel.launch @cutensornetContraction2_f64_r5r5r4(
        %av, %bv, %cv) {contraction_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]}
        : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>,
           tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %c, %result, %n0, %n1, %n2, %n3) {map = #c_flat}
        : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index)
          -> tensor<?xf64>
    %updated_memref = bufferization.to_memref %updated : memref<?xf64>
    memref.copy %updated_memref, %c_memref
        : memref<?xf64> to memref<?xf64>
    return
  }
}
