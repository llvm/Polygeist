#include <stdio.h>
#include <stdlib.h>

// DAXPY: Constant times a vector plus a vector
// y = alpha * x + y
// x: vector of length N with stride incx
// y: vector of length N with stride incy (modified in place)
// alpha: scaling factor
void daxpy(int N, double alpha, const double* x, int incx, double* y, int incy) {
    for (int i = 0; i < N; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}

// Simple version (stride = 1)
void simple_daxpy(int N, double alpha, const double* x, double* y) {
    for (int i = 0; i < N; i++) {
        y[i] += alpha * x[i];
    }
}

// Single precision version
void saxpy(int N, float alpha, const float* x, int incx, float* y, int incy) {
    for (int i = 0; i < N; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}

void print_vector(const double* x, int N, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < N; i++) {
        printf("%.2f", x[i]);
        if (i < N - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    const int N = 5;
    const double alpha = 2.0;
    
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    
    printf("AXPY Test: y = alpha * x + y\n");
    printf("alpha = %.2f\n", alpha);
    print_vector(x, N, "x");
    print_vector(y, N, "y (before)");
    
    // Apply axpy
    simple_daxpy(N, alpha, x, y);
    
    print_vector(y, N, "y (after)");
    
    printf("\nManual verification:\n");
    printf("y[0] = 2.0*1.0 + 10.0 = 12.00\n");
    printf("y[1] = 2.0*2.0 + 20.0 = 24.00\n");
    printf("y[2] = 2.0*3.0 + 30.0 = 36.00\n");
    printf("y[3] = 2.0*4.0 + 40.0 = 48.00\n");
    printf("y[4] = 2.0*5.0 + 50.0 = 60.00\n");
    
    // Test with stride
    printf("\n\nTesting with stride=2:\n");
    double x2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y2[] = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0};
    
    printf("x: [1, 2, 3, 4, 5, 6]\n");
    printf("y (before): [100, 200, 300, 400, 500, 600]\n");
    printf("Computing: y[::2] += 10.0 * x[::2]\n");
    
    daxpy(3, 10.0, x2, 2, y2, 2); // y[0,2,4] += 10*x[0,2,4]
    
    printf("y (after): [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           y2[0], y2[1], y2[2], y2[3], y2[4], y2[5]);
    printf("Expected: [110.0, 200.0, 330.0, 400.0, 550.0, 600.0]\n");
    
    return 0;
}
