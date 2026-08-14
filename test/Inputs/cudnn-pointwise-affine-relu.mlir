#map = affine_map<(d0) -> (d0)>
module {
  func.func @pointwise_affine_relu(
      %x: tensor<?xf32>, %bias: tensor<?xf32>, %out: tensor<?xf32>,
      %alpha: f32) -> tensor<?xf32> {
    %zero = arith.constant 0.0 : f32
    %r = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%x, %bias : tensor<?xf32>, tensor<?xf32>)
        outs(%out : tensor<?xf32>) {
      ^bb0(%xi: f32, %bi: f32, %out_elem: f32):
        %scaled = arith.mulf %alpha, %xi : f32
        %affine = arith.addf %scaled, %bi : f32
        %positive = arith.cmpf ogt, %affine, %zero : f32
        %activated = arith.select %positive, %affine, %zero : f32
        linalg.yield %activated : f32
    } -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}
