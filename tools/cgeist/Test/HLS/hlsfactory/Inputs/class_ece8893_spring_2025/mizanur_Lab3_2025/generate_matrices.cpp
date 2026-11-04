#include "dcl.h"

using namespace std;


// Function to generate a random sparse matrix in CSR format
void generate_sparse_matrix_csr(data_t values[], int column_indices[], int row_ptr[], int *nnz, int rows, int cols) {
    int count = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < SPARSITY) {
                values[count] = (data_t)(2.0 * (float)rand() / RAND_MAX - 1.0); // Random non-zero value
                column_indices[count] = j;
                count++;
            }
        }
        row_ptr[i + 1] = count;
    }

    *nnz = count;
}

// Function to generate a random sparse matrix in CSC format
void generate_sparse_matrix_csc(data_t values[], int row_indices[], int col_ptr[], int *nnz, int rows, int cols) {
    int count = 0;
    col_ptr[0] = 0;

    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if ((float)rand() / RAND_MAX < SPARSITY) {
                values[count] = (data_t)(2.0 * (float)rand() / RAND_MAX - 1.0); // Random non-zero value
                row_indices[count] = i;
                count++;
            }
        }
        col_ptr[j + 1] = count;
    }

    *nnz = count;
}

// Utility function to save sparse matrix to a file
void save_sparse_matrix(const char *filename, data_t values[], int indices[], int ptr[], int nnz, int size) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    fwrite(&nnz, sizeof(int), 1, file);
    fwrite(values, sizeof(data_t), nnz, file);
    fwrite(indices, sizeof(int), nnz, file);
    fwrite(ptr, sizeof(int), size + 1, file);

    fclose(file);
}

int main() {
    srand(time(NULL));

    // Matrix A (CSR format)
    data_t values_A[N * M];
    int column_indices_A[N * M];
    int row_ptr_A[(N + 1)];
    int nnz_A;

    // Matrix B (CSC format)
    data_t values_B[M * K];
    int row_indices_B[M * K];
    int col_ptr_B[(M + 1)];
    int nnz_B;

    // Generate sparse matrices
    generate_sparse_matrix_csr(values_A, column_indices_A, row_ptr_A, &nnz_A, N, M);
    generate_sparse_matrix_csc(values_B, row_indices_B, col_ptr_B, &nnz_B, M, K);

    // Save matrices to files
    char filename_A[50];
    snprintf(filename_A, sizeof(filename_A), "A_matrix_csr_sparsity_%.2f.bin", SPARSITY);
    save_sparse_matrix(filename_A, values_A, column_indices_A, row_ptr_A, nnz_A, N);
    char filename_B[50];
    snprintf(filename_B, sizeof(filename_B), "B_matrix_csc_sparsity_%.2f.bin", SPARSITY);
    save_sparse_matrix(filename_B, values_B, row_indices_B, col_ptr_B, nnz_B, M);


    printf("Matrix A (CSR) and Matrix B (CSC) with sparsity %.2f generated and saved to files.\n", SPARSITY);

    return 0;
}