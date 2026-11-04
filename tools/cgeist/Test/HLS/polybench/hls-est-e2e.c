// The PolyBench kernels of this directory, end to end through the
// hls-est BRAM estimation pipeline (as used by
// ~/circt/test/hls-est/Estimation). Each cyclic factor=4 partition of a
// local N=64 array yields 4 banks; the counts follow from the pragmas.
// Generated alongside the kernel tests.
// REQUIRES: hls-est

// RUN: cgeist %S/bicg.c --function=bicg %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.bicg.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.bicg.mlir
// RUN: %hls-est %t.bicg.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=bicg --symbol-dce --bram-analysis -o %t.bicg.est.mlir | FileCheck %s --check-prefix=BICG
// RUN: FileCheck %s --check-prefix=BICG-1D < %t.bicg.est.mlir
// RUN: FileCheck %s --check-prefix=BICG-2D < %t.bicg.est.mlir
// BICG: Total BRAM:
// BICG-1D-COUNT-16: memref.alloca() {{.*}}: memref<16xf32>
// BICG-1D-NOT: memref.alloca() {{.*}}: memref<16xf32>
// BICG-2D-COUNT-4: memref.alloca() {{.*}}: memref<64x16xf32>
// BICG-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/gesummv.c --function=gesummv %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.gesummv.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.gesummv.mlir
// RUN: %hls-est %t.gesummv.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=gesummv --symbol-dce --bram-analysis -o %t.gesummv.est.mlir | FileCheck %s --check-prefix=GESUMMV
// RUN: FileCheck %s --check-prefix=GESUMMV-1D < %t.gesummv.est.mlir
// RUN: FileCheck %s --check-prefix=GESUMMV-2D < %t.gesummv.est.mlir
// GESUMMV: Total BRAM:
// GESUMMV-1D-COUNT-16: memref.alloca() {{.*}}: memref<16xf32>
// GESUMMV-1D-NOT: memref.alloca() {{.*}}: memref<16xf32>
// GESUMMV-2D-COUNT-8: memref.alloca() {{.*}}: memref<64x16xf32>
// GESUMMV-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/k2mm.c --function=k2mm %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.k2mm.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.k2mm.mlir
// RUN: %hls-est %t.k2mm.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=k2mm --symbol-dce --bram-analysis -o %t.k2mm.est.mlir | FileCheck %s --check-prefix=K2MM
// RUN: FileCheck %s --check-prefix=K2MM-2D < %t.k2mm.est.mlir
// K2MM: Total BRAM:
// K2MM-2D-COUNT-28: memref.alloca() {{.*}}: memref<64x16xf32>
// K2MM-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/k3mm.c --function=k3mm %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.k3mm.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.k3mm.mlir
// RUN: %hls-est %t.k3mm.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=k3mm --symbol-dce --bram-analysis -o %t.k3mm.est.mlir | FileCheck %s --check-prefix=K3MM
// RUN: FileCheck %s --check-prefix=K3MM-2D < %t.k3mm.est.mlir
// K3MM: Total BRAM:
// K3MM-2D-COUNT-28: memref.alloca() {{.*}}: memref<64x16xf32>
// K3MM-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/mvt.c --function=mvt %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.mvt.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.mvt.mlir
// RUN: %hls-est %t.mvt.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=mvt --symbol-dce --bram-analysis -o %t.mvt.est.mlir | FileCheck %s --check-prefix=MVT
// RUN: FileCheck %s --check-prefix=MVT-1D < %t.mvt.est.mlir
// RUN: FileCheck %s --check-prefix=MVT-2D < %t.mvt.est.mlir
// MVT: Total BRAM:
// MVT-1D-COUNT-16: memref.alloca() {{.*}}: memref<16xf32>
// MVT-1D-NOT: memref.alloca() {{.*}}: memref<16xf32>
// MVT-2D-COUNT-4: memref.alloca() {{.*}}: memref<64x16xf32>
// MVT-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/syr2k.c --function=syr2k %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.syr2k.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.syr2k.mlir
// RUN: %hls-est %t.syr2k.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=syr2k --symbol-dce --bram-analysis -o %t.syr2k.est.mlir | FileCheck %s --check-prefix=SYR2K
// RUN: FileCheck %s --check-prefix=SYR2K-2D < %t.syr2k.est.mlir
// SYR2K: Total BRAM:
// SYR2K-2D-COUNT-28: memref.alloca() {{.*}}: memref<64x16xf32>
// SYR2K-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>

// RUN: cgeist %S/syrk.c --function=syrk %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.syrk.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.syrk.mlir
// RUN: %hls-est %t.syrk.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --symbol-privatize=exclude=syrk --symbol-dce --bram-analysis -o %t.syrk.est.mlir | FileCheck %s --check-prefix=SYRK
// RUN: FileCheck %s --check-prefix=SYRK-2D < %t.syrk.est.mlir
// SYRK: Total BRAM:
// SYRK-2D-COUNT-16: memref.alloca() {{.*}}: memref<64x16xf32>
// SYRK-2D-NOT: memref.alloca() {{.*}}: memref<64x16xf32>
