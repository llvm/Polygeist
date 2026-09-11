// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched
// RUN: FileCheck %s --check-prefix=MATCH < %t.matched
// RUN: polygeist-opt --lower-kernel-launch-to-cublas %t.matched | FileCheck %s --check-prefix=LOWER

#flat = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#scalar = affine_map<(d0)[s0] -> (s0)>
#id = affine_map<(d0) -> (d0)>

module {
  kernel.defn @cublasSgemm_flat_colmajor_nt_alpha_beta(
      %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
      %M: index, %N: index, %K: index,
      %lda: index, %ldb: index, %ldc: index,
      %beta: f32, %alpha: f32) {
    kernel.yield
  }

  func.func @source_faithful_sgemm(
      %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
      %M: index, %N: index, %K: index,
      %lda: index, %ldb: index, %ldc: index,
      %beta: f32, %alpha: f32) {
    %zero = arith.constant 0.0 : f32
    affine.for %j = 0 to %M {
      %dot = memref.alloca(%N) : memref<?xf32>
      affine.for %i = 0 to %N {
        affine.store %zero, %dot[%i] : memref<?xf32>
        %av = polygeist.submap(%A, %j, %lda, %K) {map = #flat}
            : (memref<?xf32>, index, index, index) -> memref<?xf32>
        %bv = polygeist.submap(%B, %i, %ldb, %K) {map = #flat}
            : (memref<?xf32>, index, index, index) -> memref<?xf32>
        %dv = polygeist.submap(%dot, %i, %K) {map = #scalar}
            : (memref<?xf32>, index, index) -> memref<?xf32>
        linalg.generic {
            indexing_maps = [#id, #id, #id],
            iterator_types = ["reduction"]}
            ins(%av, %bv : memref<?xf32>, memref<?xf32>)
            outs(%dv : memref<?xf32>) {
        ^bb0(%a: f32, %b: f32, %out: f32):
          %product = arith.mulf %a, %b : f32
          %sum = arith.addf %out, %product : f32
          linalg.yield %sum : f32
        }
        %d = affine.load %dot[%i] : memref<?xf32>
        %old = affine.load %C[%j + %i * symbol(%ldc)] : memref<?xf32>
        %old_scaled = arith.mulf %old, %beta : f32
        %dot_scaled = arith.mulf %alpha, %d : f32
        %new = arith.addf %old_scaled, %dot_scaled : f32
        affine.store %new, %C[%j + %i * symbol(%ldc)] : memref<?xf32>
      }
    }
    return
  }
}

// MATCH-LABEL: func.func @source_faithful_sgemm
// MATCH: kernel.launch @cublasSgemm_flat_colmajor_nt_alpha_beta
// MATCH-NOT: affine.for
// MATCH-NOT: linalg.generic
// MATCH: return
// LOWER-LABEL: func.func @source_faithful_sgemm
// LOWER: call @polygeist_cublas_sgemm_transpose
// LOWER-NOT: kernel.launch
// LOWER-NOT: affine.for
// LOWER: return
