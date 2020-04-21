// RUN: mlir-pet %S/Inputs/array.c | mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e scop_entry -entry-point-result=void -shared-libs=/home/parallels/llvm-project/build/lib/libmlir_runner_utils.so | FileCheck %s --check-prefix=A

// A: Unranked Memref rank = 1 descriptor@ = {{.*}}
// A: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data = 
// A: [28]

// RUN: mlir-pet %S/Inputs/postInc.c | mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e scop_entry -entry-point-result=void -shared-libs=/home/parallels/llvm-project/build/lib/libmlir_runner_utils.so | FileCheck %s --check-prefix=B

// B: Unranked Memref rank = 1 descriptor@ = {{.*}}
// B: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data = 
// B: [1024]

// RUN: mlir-pet %S/Inputs/loop.c | mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e scop_entry -entry-point-result=void -shared-libs=/home/parallels/llvm-project/build/lib/libmlir_runner_utils.so | FileCheck %s --check-prefix=C

// C: Unranked Memref rank = 1 descriptor@ = {{.*}}
// C: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data = 
// C: [500]

// RUN: mlir-pet %S/Inputs/scalar.c | mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e scop_entry -entry-point-result=void -shared-libs=/home/parallels/llvm-project/build/lib/libmlir_runner_utils.so | FileCheck %s --check-prefix=D

// D: Unranked Memref rank = 1 descriptor@ = {{.*}}
// D: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data = 
// D: [200]

