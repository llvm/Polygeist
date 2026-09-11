// Source-faithful compute loop from Parboil spmv/src/cpu/main.c.  Dataset
// conversion, file I/O, timers, and the outer benchmark repetition are not
// part of this callable core.
void parboil_spmv_jds_core(int dim, int depth, int len,
                            int h_nzcnt[dim], int h_ptr[depth],
                            int h_indices[len], float h_data[len],
                            float h_x_vector[dim], int h_perm[dim],
                            float h_Ax_vector[dim]) {
  for (int i = 0; i < dim; i++) {
    float sum = 0.0f;
    int bound = h_nzcnt[i];
    for (int k = 0; k < bound; k++) {
      int j = h_ptr[k] + i;
      int in = h_indices[j];
      float d = h_data[j];
      float t = h_x_vector[in];
      sum += d * t;
    }
    h_Ax_vector[h_perm[i]] = sum;
  }
}
