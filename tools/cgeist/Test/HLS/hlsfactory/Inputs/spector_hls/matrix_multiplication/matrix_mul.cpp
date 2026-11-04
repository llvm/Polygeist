#include "matrix_mul.h"
#include "params.h"

void matrix_mul(dtype *A, int B[M * M], int C[M * M]) {

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < M; k++) {

            dtype temp = *A++;

            for (int j = 0; j < M; j++) {
                C[i * M + j] += temp.data * B[k * M + j];
            }
        }
    }
}
