//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

void basic_array_partition() {
    int arr[8];
#pragma HLS array_partition variable=arr complete
    // Simple computation to test parallel access
    for(int i = 0; i < 8; i++) {
        arr[i] += 1;
    }
}

// CHECK: %alloca = memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "arr"}], hls.preserve, polygeist.varname = "arr"}