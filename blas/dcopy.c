#include <stdio.h>
#include <stdlib.h>

// DCOPY: Copy vector x to vector y
// y = x
// x: source vector of length N with stride incx
// y: destination vector of length N with stride incy
void dcopy(int N, const double* x, int incx, double* y, int incy) {
    for (int i = 0; i < N; i++) {
        y[i * incy] = x[i * incx];
    }
}

// Simple version (stride = 1)
void simple_dcopy(int N, const double* x, double* y) {
    for (int i = 0; i < N; i++) {
        y[i] = x[i];
    }
}

// Single precision version
void scopy(int N, const float* x, int incx, float* y, int incy) {
    for (int i = 0; i < N; i++) {
        y[i * incy] = x[i * incx];
    }
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
    const int N = 5;
    
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    printf("COPY Test\n");
    print_vector(x, N, "x (source)");
    print_vector(y, N, "y (before)");
    
    // Copy x to y
    simple_dcopy(N, x, y);
    
    print_vector(y, N, "y (after)");
    
    // Verify
    printf("\nVerification: ");
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (x[i] != y[i]) {
            correct = 0;
            break;
        }
    }
    printf("%s\n", correct ? "PASS" : "FAIL");
    
    // Test with stride
    printf("\n\nTesting with stride:\n");
    double src[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    double dst[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    printf("Source: [10, 20, 30, 40, 50, 60]\n");
    printf("Copying every other element (incx=2) to every position (incy=1):\n");
    dcopy(3, src, 2, dst, 1); // Copy src[0,2,4] to dst[0,1,2]
    printf("Result: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
    printf("Expected: [10.0, 30.0, 50.0, 0.0, 0.0, 0.0]\n");
    
    return 0;
}
