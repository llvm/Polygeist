#include <stdio.h>
#include <stdlib.h>

// DSCAL: Scale a vector by a constant
// x = alpha * x
// x: vector of length N with stride incx
// alpha: scaling factor
void dscal(int N, double alpha, double* x, int incx) {
    for (int i = 0; i < N; i++) {
        x[i * incx] *= alpha;
    }
}

// Simple version (stride = 1)
void simple_dscal(int N, double alpha, double* x) {
    for (int i = 0; i < N; i++) {
        x[i] *= alpha;
    }
}

// Single precision version
void sscal(int N, float alpha, float* x, int incx) {
    for (int i = 0; i < N; i++) {
        x[i * incx] *= alpha;
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
    const double alpha = 2.5;
    
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    printf("SCAL Test\n");
    printf("alpha = %.2f\n", alpha);
    print_vector(x, N, "x (before)");
    
    // Apply scaling
    simple_dscal(N, alpha, x);
    
    print_vector(x, N, "x (after)");
    
    printf("\nManual verification:\n");
    printf("Expected: [2.50, 5.00, 7.50, 10.00, 12.50]\n");
    
    // Test with stride
    printf("\n\nTesting with stride=2:\n");
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    printf("Original: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n");
    dscal(3, 10.0, y, 2); // Scale elements at positions 0, 2, 4
    printf("After scaling every other element by 10:\n");
    printf("Result:   [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           y[0], y[1], y[2], y[3], y[4], y[5]);
    printf("Expected: [10.0, 2.0, 30.0, 4.0, 50.0, 6.0]\n");
    
    return 0;
}
