#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// DASUM: Sum of absolute values
// result = sum(|x[i]|)
// x: vector of length N with stride incx
double dasum(int N, const double* x, int incx) {
    double result = 0.0;
    
    for (int i = 0; i < N; i++) {
        result += fabs(x[i * incx]);
    }
    
    return result;
}

// Simple version (stride = 1)
double simple_dasum(int N, const double* x) {
    double result = 0.0;
    
    for (int i = 0; i < N; i++) {
        result += fabs(x[i]);
    }
    
    return result;
}

// Single precision version
float sasum(int N, const float* x, int incx) {
    float result = 0.0f;
    
    for (int i = 0; i < N; i++) {
        result += fabsf(x[i * incx]);
    }
    
    return result;
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
    const int N = 6;
    
    double x[] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    
    printf("ASUM Test: sum of absolute values\n");
    print_vector(x, N, "x");
    
    double result = simple_dasum(N, x);
    
    printf("\nasum(x) = %.1f\n", result);
    
    printf("\nManual verification:\n");
    printf("|1.0| + |-2.0| + |3.0| + |-4.0| + |5.0| + |-6.0|\n");
    printf("= 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0\n");
    printf("= 21.0\n");
    
    // Test with stride
    printf("\n\nTesting with stride=2 (every other element):\n");
    double result_stride = dasum(3, x, 2);
    printf("asum(x[::2]) = %.1f\n", result_stride);
    printf("Manual: |%.1f| + |%.1f| + |%.1f| = %.1f\n",
           x[0], x[2], x[4], fabs(x[0]) + fabs(x[2]) + fabs(x[4]));
    
    return 0;
}
