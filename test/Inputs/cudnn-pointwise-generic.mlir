#map = affine_map<(d0) -> (d0)>
module {
  func.func @pointwise_generic(
      %x: tensor<?xf32>, %y: tensor<?xf32>, %out: tensor<?xf32>,
      %scale: f32, %offset: f32) -> tensor<?xf32> {
    %r = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%x, %y : tensor<?xf32>, tensor<?xf32>)
        outs(%out : tensor<?xf32>) {
      ^bb0(%xi: f32, %yi: f32, %out_elem: f32):
        %difference = arith.subf %xi, %yi : f32
        %scaled = arith.mulf %difference, %scale : f32
        %activated = math.tanh %scaled : f32
        %shifted = arith.addf %activated, %offset : f32
        linalg.yield %shifted : f32
    } -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}
