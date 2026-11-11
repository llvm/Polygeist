// RUN: polygeist-opt %s --split-input-file --lower-kernel-launch-to-cublas --verify-diagnostics

module {
  kernel.defn @cubPrefixMaxLowerBounds_i32_memref(%n: index, %x: memref<?xi64>, %y: memref<?xi64>) { kernel.yield }
  func.func @i64(%n: index, %x: memref<?xi64>, %y: memref<?xi64>) {
    // expected-error @+1 {{cuSPARSE index conversion: expected index rows and rank-1 i32 input/output memrefs}}
    kernel.launch @cubPrefixMaxLowerBounds_i32_memref(%n, %x, %y) : (index, memref<?xi64>, memref<?xi64>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cubPrefixMaxLowerBounds_i32_memref(%n: index, %x: memref<?x?xi32>, %y: memref<?xi32>) { kernel.yield }
  func.func @rank(%n: index, %x: memref<?x?xi32>, %y: memref<?xi32>) {
    // expected-error @+1 {{cuSPARSE index conversion: expected index rows and rank-1 i32 input/output memrefs}}
    kernel.launch @cubPrefixMaxLowerBounds_i32_memref(%n, %x, %y) : (index, memref<?x?xi32>, memref<?xi32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cubPrefixMaxLowerBounds_i32_memref(%n: i32, %x: memref<?xi32>, %y: memref<?xi32>) { kernel.yield }
  func.func @scalar(%n: i32, %x: memref<?xi32>, %y: memref<?xi32>) {
    // expected-error @+1 {{cuSPARSE index conversion: expected index rows and rank-1 i32 input/output memrefs}}
    kernel.launch @cubPrefixMaxLowerBounds_i32_memref(%n, %x, %y) : (i32, memref<?xi32>, memref<?xi32>) -> ()
    return
  }
}

// -----

module {
  kernel.defn @cubPrefixMaxLowerBounds_i32_memref(%n: index, %x: memref<?xi32, strided<[2]>>, %y: memref<?xi32>) { kernel.yield }
  func.func @stride(%n: index, %x: memref<?xi32, strided<[2]>>, %y: memref<?xi32>) {
    // expected-error @+1 {{CUB prefix lower bounds requires contiguous input/output memrefs}}
    kernel.launch @cubPrefixMaxLowerBounds_i32_memref(%n, %x, %y) : (index, memref<?xi32, strided<[2]>>, memref<?xi32>) -> ()
    return
  }
}
