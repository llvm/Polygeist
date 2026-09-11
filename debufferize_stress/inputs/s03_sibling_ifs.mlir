// INTENT: two sibling scf.if at function-body level, both storing into the
// same memref. The second if must see the FIRST if's *rebuilt* result as its
// entry tensor, not the original to_tensor — the function-body if eager
// rebuild at line 942-967 is what makes this work. Verify.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @sibling_ifs(%c1: i1, %c2: i1, %x: memref<8xf64>) {
    %v1 = arith.constant 1.0 : f64
    %v2 = arith.constant 2.0 : f64
    %i0 = arith.constant 0 : index
    %i1_ = arith.constant 1 : index
    scf.if %c1 {
      memref.store %v1, %x[%i0] : memref<8xf64>
    }
    scf.if %c2 {
      memref.store %v2, %x[%i1_] : memref<8xf64>
    }
    return
  }
}
