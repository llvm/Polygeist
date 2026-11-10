#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s NX NY NZ OUTPUT\n", argv[0]);
    return 2;
  }
  int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
  if (nx <= 0 || ny <= 0 || nz <= 0)
    return 2;
  FILE *file = fopen(argv[4], "wb");
  if (!file)
    return 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        float value = (float)(((i * 13 + j * 7 + k * 3) % 37) - 18) / 19.0f;
        if (fwrite(&value, sizeof(value), 1, file) != 1)
          return 1;
      }
  return fclose(file) != 0;
}
