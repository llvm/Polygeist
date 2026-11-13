// RUN: polygeist-opt --select-contraction-backend='allow-tf32=true target-arch=sm_87' %s | FileCheck %s
// RUN: polygeist-opt --select-contraction-backend %s | FileCheck %s --check-prefix=STRICT

module {
  // Arbitrary post-recognition opcode names demonstrate that neutral semantic
  // metadata, not an original source/function name, drives these decisions.
  kernel.defn @renamed_a(%A: tensor<7x4xf32>, %B: tensor<4x5xf32>,
                         %C: tensor<7x5xf32>) -> tensor<7x5xf32> {
    kernel.yield %C : tensor<7x5xf32>
  }
  kernel.defn @renamed_b(%A: tensor<31x9xf32>, %B: tensor<9x17xf32>,
                         %C: tensor<31x17xf32>) -> tensor<31x17xf32> {
    kernel.yield %C : tensor<31x17xf32>
  }
  kernel.defn @renamed_c(%A: tensor<64x32xf32>, %B: tensor<32x64xf32>,
                         %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
    kernel.yield %C : tensor<64x64xf32>
  }
  kernel.defn @renamed_d(%A: tensor<16x8xf32>, %B: tensor<8x16xf32>,
                         %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    kernel.yield %C : tensor<16x16xf32>
  }
  kernel.defn @cublasSgemm_nt(%A: tensor<16x8xf32>,
                              %B: tensor<16x8xf32>,
                              %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    kernel.yield %C : tensor<16x16xf32>
  }
  kernel.defn @rankk_op(%A: tensor<?x?xf32>, %C: tensor<?x?xf32>,
                        %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }

  func.func @test(%a0: tensor<7x4xf32>, %b0: tensor<4x5xf32>,
                  %c0: tensor<7x5xf32>,
                  %a1: tensor<31x9xf32>, %b1: tensor<9x17xf32>,
                  %c1: tensor<31x17xf32>,
                  %a2: tensor<64x32xf32>, %b2: tensor<32x64xf32>,
                  %c2: tensor<64x64xf32>,
                  %a3: tensor<16x8xf32>, %b3: tensor<8x16xf32>,
                  %bt3: tensor<16x8xf32>, %c3: tensor<16x16xf32>,
                  %ra: tensor<?x?xf32>, %rc: tensor<?x?xf32>,
                  %beta: f32, %alpha: f32) {
    %0 = kernel.launch @renamed_a(%a0, %b0, %c0) {
      polygeist.contraction.kind = "gemm",
      polygeist.contraction.m = 7 : index,
      polygeist.contraction.n = 5 : index,
      polygeist.contraction.k = 4 : index
    } : (tensor<7x4xf32>, tensor<4x5xf32>, tensor<7x5xf32>) -> tensor<7x5xf32>
    %1 = kernel.launch @renamed_b(%a1, %b1, %c1) {
      polygeist.contraction.kind = "gemm",
      polygeist.contraction.m = 31 : index,
      polygeist.contraction.n = 17 : index,
      polygeist.contraction.k = 9 : index
    } : (tensor<31x9xf32>, tensor<9x17xf32>, tensor<31x17xf32>) -> tensor<31x17xf32>
    %2 = kernel.launch @renamed_c(%a2, %b2, %c2) {
      polygeist.contraction.kind = "contraction",
      polygeist.contraction.cublasdx_legal,
      polygeist.contraction.m = 64 : index,
      polygeist.contraction.n = 64 : index,
      polygeist.contraction.k = 32 : index
    } : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = kernel.launch @renamed_d(%a3, %b3, %c3) {
      polygeist.contraction.kind = "gemm",
      polygeist.contraction.m = 16 : index,
      polygeist.contraction.n = 16 : index,
      polygeist.contraction.k = 8 : index
    } : (tensor<16x8xf32>, tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    // Existing semantic ISA symbols remain supported while the matcher moves
    // to neutral contraction metadata. Here NT means physical B is N x K.
    %4 = kernel.launch @cublasSgemm_nt(%a3, %bt3, %c3) :
      (tensor<16x8xf32>, tensor<16x8xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = kernel.launch @rankk_op(%ra, %rc, %beta, %alpha) {
      polygeist.contraction.kind = "syrk"
    } : (tensor<?x?xf32>, tensor<?x?xf32>, f32, f32) -> tensor<?x?xf32>
    return
  }
}

// CHECK: kernel.launch @renamed_a
// CHECK-SAME: polygeist.contraction.backend = "cuda-fma"
// CHECK: kernel.launch @renamed_b
// CHECK-SAME: polygeist.contraction.backend = "direct-mma"
// CHECK: kernel.launch @renamed_c
// CHECK-SAME: polygeist.contraction.backend = "cublasdx"
// CHECK: kernel.launch @renamed_d
// CHECK-SAME: polygeist.contraction.backend = "wmma"
// CHECK: kernel.launch @cublasSgemm_nt
// CHECK-SAME: polygeist.contraction.backend = "wmma"
// CHECK-SAME: polygeist.contraction.k = 8 : index
// CHECK: kernel.launch @rankk_op
// CHECK-SAME: polygeist.contraction.backend = "vendor-library"
// CHECK-SAME: polygeist.contraction.selection_reason = "no-verified-rankk-tf32-benefit"

// STRICT: kernel.launch @renamed_a
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
// STRICT: kernel.launch @renamed_b
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
// STRICT: kernel.launch @renamed_c
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
// STRICT: kernel.launch @renamed_d
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
// STRICT: kernel.launch @cublasSgemm_nt
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
// STRICT: kernel.launch @rankk_op
// STRICT-SAME: polygeist.contraction.backend = "vendor-library"
