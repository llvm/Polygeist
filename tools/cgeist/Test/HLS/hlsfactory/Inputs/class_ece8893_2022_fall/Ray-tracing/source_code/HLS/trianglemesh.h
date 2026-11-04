#ifndef _TRIANGLE_MESH_
#define _TRIANGLE_MESH_

#include <stdint.h>
#include "common.h"

typedef struct triangle_mesh
{
    uint32_t numTris;                         // number of triangles
    fixed_t P[MAX_VERT_INDEX][3];               // triangles vertex position
    uint32_t trisIndex[NUM_TRIS * 3];         // vertex index array
    fixed_t N[NUM_TRIS * 3][3];                 // triangles vertex normals
    fixed_t texCoordinates[NUM_TRIS * 3][2];    // triangles texture coordinates
    fixed_t objectToWorld[4][4], worldToObject[4][4];
} triangle_mesh_t;

void render(
    fixed_t P1_DRAM[NUM_TRIS][3],
    fixed_t P2_DRAM[NUM_TRIS][3],
    fixed_t P3_DRAM[NUM_TRIS][3],
    fixed_t texCoordinates[NUM_TRIS * 3][2],
    fixed_t framebuffer[WIDTH * HEIGHT][3],
    fixed_t cameraToWorld_DRAM[4][4],
    fixed_t backgroundColor[3],
    fixed_t frame_width,
    fixed_t frame_height,
    fixed_t frame_scale);

#endif

