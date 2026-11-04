//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

template <typename DT>
DT foo(DT a, DT b) {
    return a + b;
}

// ------------------------------------------------------------
// Top-level function that calls foo<int> and foo<float>
// ------------------------------------------------------------
void top(int *out_int, float *out_float, int a, int b, float x, float y) {

#pragma HLS ALLOCATION function instances=foo<int> limit=1
#pragma HLS ALLOCATION function instances=foo<float> limit=1

    int r1 = foo<int>(a, b);
    float r2 = foo<float>(x, y);

    *out_int = r1;
    *out_float = r2;
}

// CHECK: hls.allocation = [{instances = ["foo<int>"], limit = 1 : i32, type = "function"}, {instances = ["foo<float>"], limit = 1 : i32, type = "function"}]