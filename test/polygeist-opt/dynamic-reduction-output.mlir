// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s
// Each parallel dimension maps injectively to a translated output coordinate.
// The two reduction dimensions must disappear from the physical output view.
// CHECK-LABEL: func.func @translated_dynamic_reduction
// CHECK-NOT: polygeist.submapInverse
// CHECK: tensor.insert_slice
// CHECK-NOT: polygeist.submapInverse
func.func @translated_dynamic_reduction(%input: tensor<?x?x3x3xi8>, %out: tensor<?x?xi8>, %h: index, %w: index) -> tensor<?x?xi8> {
  %three = arith.constant 3 : index
  %view = polygeist.submap(%out, %h, %w, %three, %three) {map = affine_map<(i,j,k,l)->(i+1,j+1)>} : (tensor<?x?xi8>,index,index,index,index)->tensor<?x?x?x?xi8>
  %r = linalg.generic {indexing_maps = [affine_map<(i,j,k,l)->(i,j,k,l)>,affine_map<(i,j,k,l)->(i,j,k,l)>], iterator_types = ["parallel","parallel","reduction","reduction"]} ins(%input:tensor<?x?x3x3xi8>) outs(%view:tensor<?x?x?x?xi8>) {
  ^bb0(%x:i8,%y:i8):
    %z=arith.maxui %x,%y:i8
    linalg.yield %z:i8
  } -> tensor<?x?x?x?xi8>
  %restored = polygeist.submapInverse(%out, %r, %h, %w, %three, %three) {map = affine_map<(i,j,k,l)->(i+1,j+1)>} : (tensor<?x?xi8>,tensor<?x?x?x?xi8>,index,index,index,index)->tensor<?x?xi8>
  return %restored:tensor<?x?xi8>
}
