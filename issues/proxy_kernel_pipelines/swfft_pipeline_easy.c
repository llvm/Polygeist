// SWFFT-style local redistribution and transpose fixture designed to raise
// cleanly. It models local pack/unpack and slab/transpose movement without MPI.

#define SX 8
#define SY 8
#define SZ 8

void swfft_pipeline_easy(const double pencil_in[SX * SY * SZ],
                         const double cube_in[SX][SY][SZ],
                         double chunk[SX][SY][SZ],
                         double slab[SX][SY],
                         double cube_work[SX][SY][SZ],
                         double xy[SY][SX][SZ],
                         double yz[SX][SZ][SY],
                         double pencil_out[SX * SY * SZ]) {
  // Redistribute 2D pencil to 3D chunk.
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        chunk[x][y][z] = pencil_in[(x * SY + y) * SZ + z];

  // Copy one local slab at a fixed z-plane.
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      slab[x][y] = chunk[x][y][3];

  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      cube_work[x][y][3] = slab[x][y];

  // Local transposes.
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        xy[y][x][z] = cube_in[x][y][z];

  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        yz[x][z][y] = cube_in[x][y][z];

  // Redistribute 3D chunk back to 2D pencil.
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        pencil_out[(x * SY + y) * SZ + z] = chunk[x][y][z];
}
