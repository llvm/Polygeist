#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GEMM: C = alpha * A * B + beta * C
// A: M x K matrix with leading dimension LDA
// B: K x N matrix with leading dimension LDB  
// C: M x N matrix with leading dimension LDC
void dgemm(char transa, char transb, int M, int N, int K,
           double alpha,
           const double* A, int LDA,
           const double* B, int LDB,
           double beta,
           double* C, int LDC) {
    
    // Handle beta scaling first
    if (beta == 0.0) {
        // Zero out C
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * LDC + j] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        // Scale C by beta
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * LDC + j] *= beta;
            }
        }
    }
    
    // Early return if alpha is zero
    if (alpha == 0.0) {
        return;
    }
    
    // Handle different transpose cases
    if (transa == 'N' && transb == 'N') {
        // C = alpha * A * B + beta * C (no transpose)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * LDA + k] * B[k * LDB + j];
                }
                C[i * LDC + j] += alpha * sum;
            }
        }
    } else if (transa == 'T' && transb == 'N') {
        // C = alpha * A^T * B + beta * C
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[k * LDA + i] * B[k * LDB + j];
                }
                C[i * LDC + j] += alpha * sum;
            }
        }
    } else if (transa == 'N' && transb == 'T') {
        // C = alpha * A * B^T + beta * C
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * LDA + k] * B[j * LDB + k];
                }
                C[i * LDC + j] += alpha * sum;
            }
        }
    } else if (transa == 'T' && transb == 'T') {
        // C = alpha * A^T * B^T + beta * C
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[k * LDA + i] * B[j * LDB + k];
                }
                C[i * LDC + j] += alpha * sum;
            }
        }
    }
}

// Simple GEMM (no transpose, alpha=1, beta=0)
void simple_dgemm(int M, int N, int K,
                  const double* A, int LDA,
                  const double* B, int LDB,
                  double* C, int LDC) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * LDA + k] * B[k * LDB + j];
            }
            C[i * LDC + j] = sum;
        }
    }
}

// Single precision version
void sgemm(char transa, char transb, int M, int N, int K,
           float alpha,
           const float* A, int LDA,
           const float* B, int LDB,
           float beta,
           float* C, int LDC) {
    
    // Handle beta scaling
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * LDC + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * LDC + j] *= beta;
            }
        }
    }
    
    if (alpha == 0.0f) return;
    
    // Only implement N,N case for simplicity
    if (transa == 'N' && transb == 'N') {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * LDA + k] * B[k * LDB + j];
                }
                C[i * LDC + j] += alpha * sum;
            }
        }
    }
}

// Utility functions
void print_matrix(const double* matrix, int rows, int cols, int LD, const char* name) {
    printf("%s (%dx%d with LD=%d):\n", name, rows, cols, LD);
    for (int i = 0; i < rows; i++) {
        printf("Row %d: [", i);
        for (int j = 0; j < cols; j++) {
            printf("%8.3f", matrix[i * LD + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}
