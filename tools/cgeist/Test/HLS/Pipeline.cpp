//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s

// The loop must have an observable effect: with `j` a by-value parameter the
// loop was dead and is (correctly) removed, leaving nothing to annotate.
void my_kernel(int *j) {
    *j = 0;
    for(int i = 0; i < 10; ++i) {
        #pragma HLS pipeline II=1
        j[i] += 1;
    }
}

// CHECK-NOT: warning
// CHECK: affine.for
// CHECK: } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
