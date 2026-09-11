#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// DNRM2: Euclidean norm (L2 norm) of a vector
// result = sqrt(sum(x[i]^2))
// x: vector of length N with stride incx
double dnrm2(int N, const double* x, int incx) {
    double sum = 0.0;
    
    for (int i = 0; i < N; i++) {
        double val = x[i * incx];
        sum += val * val;
    }
    
    return sqrt(sum);
}

// Simple version (stride = 1)
double simple_dnrm2(int N, const double* x) {
    double sum = 0.0;
    
    for (int i = 0; i < N; i++) {
        sum += x[i] * x[i];
    }
    
    return sqrt(sum);
}

// Single precision version
float snrm2(int N, const float* x, int incx) {
    float sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        float val = x[i * incx];
        sum += val * val;
    }
    
    return sqrtf(sum);
}

void print_vector(const double* x, int N, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < N; i++) {
        printf("%.1f", x[i]);
        if (i < N - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    const int N = 4;
    
    double x[] = {3.0, 4.0, 0.0, 0.0};
    
    printf("NRM2 Test: Euclidean norm (L2 norm)\n");
    print_vector(x, N, "x");
    
    double result = simple_dnrm2(N, x);
    
    printf("\n||x||_2 = %.2f\n", result);
    
    printf("\nManual verification:\n");
    printf("sqrt(3^2 + 4^2 + 0^2 + 0^2)\n");
    printf("= sqrt(9 + 16 + 0 + 0)\n");
    printf("= sqrt(25)\n");
    printf("= 5.00\n");
    
    // Test with unit vector
    printf("\n\nTest with unit vector:\n");
    double unit[] = {1.0, 0.0, 0.0};
    print_vector(unit, 3, "unit");
    double norm_unit = simple_dnrm2(3, unit);
    printf("||unit||_2 = %.2f (expected: 1.00)\n", norm_unit);
    
    // Test with stride
    printf("\n\nTesting with stride=2:\n");
    double y[] = {3.0, 100.0, 4.0, 200.0, 0.0, 300.0};
    printf("y: [3.0, 100.0, 4.0, 200.0, 0.0, 300.0]\n");
    double result_stride = dnrm2(3, y, 2);
    printf("||y[::2]||_2 = %.2f\n", result_stride);
    printf("Manual: sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5.00\n");
    
    return 0;
}
