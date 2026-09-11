// RUN: not polygeist-opt --lower-kernel-launch-to-cublas %s 2>&1 | FileCheck %s

module {
  // Semantic recognition alone is not enough to authorize a library launch.
  // An unsupported symbol must fail explicitly instead of surviving into a
  // later pipeline or becoming a declaration for a nonexistent runtime shim.
  kernel.defn @semantic_only_kernel(%input: memref<?xf32>,
                                    %output: memref<?xf32>) {
    kernel.yield
  }

  func.func @unsupported(%input: memref<?xf32>, %output: memref<?xf32>) {
    kernel.launch @semantic_only_kernel(%input, %output)
        : (memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK: lower-kernel-launch-to-cublas: no shim ABI lowering for library symbol @semantic_only_kernel
