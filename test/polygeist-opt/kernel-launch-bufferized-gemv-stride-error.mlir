// RUN: polygeist-opt --lower-kernel-launch-to-cublas --verify-diagnostics %s

module {
  kernel.defn @cublasDgemv(%a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }
  func.func @reject_nonunit_vector(%a: memref<?x?xf64>,
      %x: memref<?xf64, strided<[2]>>, %y: memref<?xf64>) {
    // expected-error @+1 {{bufferized GEMV requires a row-major matrix and unit-stride vectors}}
    kernel.launch @cublasDgemv(%a, %x, %y) {polygeist.bufferized, polygeist.result_destinations = array<i64: 2>} : (memref<?x?xf64>, memref<?xf64, strided<[2]>>, memref<?xf64>) -> ()
    return
  }
}
