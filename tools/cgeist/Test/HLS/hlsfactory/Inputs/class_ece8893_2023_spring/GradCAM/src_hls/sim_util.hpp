#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "util.h"

float *read_bin_file(const std::string filepath, const std::vector<int> dims)
{
    uint64_t size = 1;
    for (int i = 0; i < dims.size(); i++)
        size *= dims[i];

    float *ans;
    ans = (float *) malloc(size * sizeof(float));

    std::ifstream input(filepath, std::ios::in | std::ios::binary);
    input.read((char*)(ans), size*sizeof(float));
    input.close();

    return ans;
}

template <typename fixp_t>
void write_to_file(const std::string& filename, const std::vector<int>& dims, const fixp_t* array) {
  size_t num_elements = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    num_elements *= dims[i];
  }

  std::cout << "Writing to file: " << filename << std::endl;
  const float* float_array = reinterpret_cast<const float*>(array);

  std::ofstream output_file(filename, std::ios::out | std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(float_array), num_elements * sizeof(float));

  output_file.close();
}

template <typename fixp_t, int F, int C, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[F][C][M][N], fixp_t fixp_arr[F][C][M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);

    if (input.fail()) {
        std::cout << "Failed to open file: " << filepath << std::endl;
        exit(1);
    }

    input.read((char*) (fp_arr), F*C*M*N*sizeof(float));
    input.close();

    for (int f = 0; f < F; f++) {
        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    fixp_arr[f][c][m][n] = static_cast<fm_t>(fp_arr[f][c][m][n]);
                }
            }
        }
    }
}

template <typename fixp_t, int C, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[C][M][N], fixp_t fixp_arr[C][M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);

    if (input.fail()) {
        std::cout << "Failed to open file: " << filepath << std::endl;
        exit(1);
    }

    input.read((char*) (fp_arr), C*M*N*sizeof(float));
    input.close();

    for (int c = 0; c < C; c++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                fixp_arr[c][m][n] = static_cast<fm_t>(fp_arr[c][m][n]);
            }
        }
    }
}

template <typename fixp_t, int M, int N>
void load_fp_and_fixp_vals(float fp_arr[M][N], fixp_t fixp_arr[M][N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);

    if (input.fail()) {
        std::cout << "Failed to open file: " << filepath << std::endl;
        exit(1);
    }

    input.read((char*) (fp_arr), M*N*sizeof(float));
    input.close();

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            fixp_arr[m][n] = static_cast<fm_t>(fp_arr[m][n]);
        }
    }
}

template <typename fixp_t, int N>
void load_fp_and_fixp_vals(float fp_arr[N], fixp_t fixp_arr[N], const std::string filepath){

    std::ifstream input(filepath, std::ios::in | std::ios::binary);

    if (input.fail()) {
        std::cout << "Failed to open file: " << filepath << std::endl;
        exit(1);
    }

    input.read((char*) (fp_arr), N*sizeof(float));
    input.close();

    for (int n = 0; n < N; n++) {
        fixp_arr[n] = static_cast<fm_t>(fp_arr[n]);
    }
}


