//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

void fir_filter(int x[16], int h[8], int y[16]) {
    for (int i = 0; i < 16; i++) {
#pragma HLS unroll
        int acc = 0;
        for (int j = 0; j < 8; j++)
            acc += x[i-j] * h[j];
        y[i] = acc;
    }
}

// CHECK: {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}