// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @customJdsSpmv_f32_memref(
      %rows: index, %nzcnt: memref<?xi32>, %ptr: memref<?xi32>,
      %indices: memref<?xi32>, %data: memref<?xf32>, %x: memref<?xf32>,
      %perm: memref<?xi32>, %out: memref<?xf32>) { kernel.yield }
  kernel.defn @customCsrSpmv_f64_memref(
      %rows: index, %rowptr: memref<101xi32>, %cols: memref<3600xi32>,
      %data: memref<3600xf64>, %x: memref<102xf64>,
      %out: memref<102xf64>) { kernel.yield }

  func.func @sparse(%rows: index,
      %nzcnt: memref<?xi32>, %ptr: memref<?xi32>, %ji: memref<?xi32>,
      %jd: memref<?xf32>, %jx: memref<?xf32>, %perm: memref<?xi32>,
      %jo: memref<?xf32>, %rp: memref<101xi32>, %ci: memref<3600xi32>,
      %cd: memref<3600xf64>, %cx: memref<102xf64>, %co: memref<102xf64>) {
    kernel.launch @customJdsSpmv_f32_memref(
        %rows, %nzcnt, %ptr, %ji, %jd, %jx, %perm, %jo)
        : (index, memref<?xi32>, memref<?xi32>, memref<?xi32>,
           memref<?xf32>, memref<?xf32>, memref<?xi32>, memref<?xf32>) -> ()
    kernel.launch @customCsrSpmv_f64_memref(%rows, %rp, %ci, %cd, %cx, %co)
        : (index, memref<101xi32>, memref<3600xi32>, memref<3600xf64>,
           memref<102xf64>, memref<102xf64>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @sparse
// CHECK: call @polygeist_jds_spmv_f32_sized
// CHECK: call @polygeist_csr_spmv_f64_sized
// CHECK-NOT: kernel.launch
