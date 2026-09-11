// RUN: clang -O2 -DPOLYGEIST_CPU_USE_CBLAS -I%polygeist_src_root/runtime -c %polygeist_src_root/runtime/polygeist_cublas_rt_cpu.c -o %t

// Enabling CBLAS includes <complex.h>, whose standard `I` macro must not
// collide with runtime parameter names.
