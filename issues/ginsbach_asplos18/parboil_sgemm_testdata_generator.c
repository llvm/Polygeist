#include <stdio.h>
#include <stdlib.h>

static int write_matrix(const char *path, int rows, int cols, int seed) {
  FILE *file = fopen(path, "w");
  if (!file)
    return 1;
  fprintf(file, "%d %d", rows, cols);
  for (int col = 0; col < cols; ++col)
    for (int row = 0; row < rows; ++row) {
      int value = ((row * 17 + col * 29 + seed) % 31) - 15;
      fprintf(file, " %.8g", (float)value / 16.0f);
    }
  fputc('\n', file);
  return fclose(file) != 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s SIZE A UNUSED BT\n", argv[0]);
    return 2;
  }
  int size = atoi(argv[1]);
  if (size <= 0)
    return 2;
  return write_matrix(argv[2], size, size, 3) ||
         write_matrix(argv[3], 1, 1, 0) ||
         write_matrix(argv[4], size, size, 11);
}
