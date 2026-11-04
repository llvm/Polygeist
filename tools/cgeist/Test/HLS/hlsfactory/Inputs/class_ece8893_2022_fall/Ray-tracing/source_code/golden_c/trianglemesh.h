#ifndef _TRIANGLE_MESH_
#define _TRIANGLE_MESH_

#include <stdint.h>
#include "common.h"

typedef struct triangle_mesh
{
    uint32_t numTris;                         // number of triangles
    float P[MAX_VERT_INDEX][3];               // triangles vertex position
    uint32_t trisIndex[NUM_TRIS * 3];         // vertex index array
    float N[NUM_TRIS * 3][3];                 // triangles vertex normals
    float texCoordinates[NUM_TRIS * 3][2];    // triangles texture coordinates
    float objectToWorld[4][4], worldToObject[4][4];
} triangle_mesh_t;

void render(
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float texCoordinates[NUM_TRIS * 3][2],
    float framebuffer[WIDTH * HEIGHT][3],
    float cameraToWorld[4][4],
    float backgroundColor[3]);

#endif

