#include <stdio.h>
#include <stdlib.h>

// DDOT: Compute dot product of two vectors
// result = sum(x[i] * y[i])
// x: vector of length N with stride incx
// y: vector of length N with stride incy
double ddot(int N, const double* x, int incx, const double* y, int incy) {
    double result = 0.0;
    
    for (int i = 0; i < N; i++) {
        result += x[i * incx] * y[i * incy];
    }
    
    return result;
}

// Simple version (stride = 1)
double simple_ddot(int N, const double* x, const double* y) {
    double result = 0.0;
    
    for (int i = 0; i < N; i++) {
        result += x[i] * y[i];
    }
    
    return result;
}

// Single precision version
float sdot(int N, const float* x, int incx, const float* y, int incy) {
    float result = 0.0f;
    
    for (int i = 0; i < N; i++) {
        result += x[i * incx] * y[i * incy];
    }
    
    return result;
}

int main() {
    const int N = 5;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    
    printf("DOT Product Test\n");
    printf("x: [");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", x[i]);
    }
    printf("]\n");
    
    printf("y: [");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", y[i]);
    }
    printf("]\n\n");
    
    // Test simple version
    double result = simple_ddot(N, x, y);
    printf("dot(x, y) = %.1f\n", result);
    
    // Manual verification
    double manual = 0.0;
    for (int i = 0; i < N; i++) {
        manual += x[i] * y[i];
        printf("  %.1f * %.1f = %.1f\n", x[i], y[i], x[i] * y[i]);
    }
    printf("Expected: %.1f, Actual: %.1f\n\n", manual, result);
    
    // Test with stride
    printf("Testing with stride=2 (every other element):\n");
    double result_stride = ddot(3, x, 2, y, 2);
    printf("dot(x[::2], y[::2]) = %.1f\n", result_stride);
    printf("Manual: %.1f*%.1f + %.1f*%.1f + %.1f*%.1f = %.1f\n",
           x[0], y[0], x[2], y[2], x[4], y[4],
           x[0]*y[0] + x[2]*y[2] + x[4]*y[4]);
    
    return 0;
}
