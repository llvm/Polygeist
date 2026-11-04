//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

void mac_top(int input[64], int &result) {
#pragma HLS bind_storage variable=weights type=ram_1p  impl=bram
#pragma HLS bind_storage variable=bias    type=ram_1p  impl=lutram
#pragma HLS bind_storage variable=output  type=ram_2p  impl=uram  latency=3
    int weights[64];
    int bias[16];
    int output[16];

    for (int i = 0; i < 16; i++) {
        output[i] = bias[i];
        for (int j = 0; j < 4; j++)
            output[i] += input[i*4+j] * weights[i*4+j];
    }
    for (int i = 0; i < 64; i++) {
        result += output[i];
    }
}

// CHECK: %alloca = memref.alloca() {hls.bind_storage = [{impl = "uram", latency = 3 : i32, type = "ram_2p", variable = "output"}], hls.preserve, polygeist.varname = "output"} : memref<16xi32>
// CHECK: %alloca_0 = memref.alloca() {hls.bind_storage = [{impl = "lutram", type = "ram_1p", variable = "bias"}], hls.preserve, polygeist.varname = "bias"} : memref<16xi32>
// CHECK: %alloca_1 = memref.alloca() {hls.bind_storage = [{impl = "bram", type = "ram_1p", variable = "weights"}], hls.preserve, polygeist.varname = "weights"} : memref<64xi32>
