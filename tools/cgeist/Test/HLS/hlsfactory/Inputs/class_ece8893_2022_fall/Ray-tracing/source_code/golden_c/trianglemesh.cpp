#include <cmath>
#include <algorithm>

#include "trianglemesh.h"
#include "common.h"


// Using index0 = x, index1 = y, index2 = z
bool rayTriangleIntersect(
    float orig[3], float dir[3],
    float v0[3], float v1[3], float v2[3],
    float &t, float &u, float &v)
{

    // v0v1 = v1 - v0;
    float v0v1[3];
    customSubtract(v1, v0, v0v1);

    // v0v2 = v2 - v0;
    float v0v2[3];
    customSubtract(v2, v0, v0v2);

    // pvec = dir x v0v2;
    float pvec[3];
    customCrossProduct(dir, v0v2, pvec);

    // det = v0v1.pvec;
    float det;
    customDotProduct(v0v1, pvec, det);

    float detTest = det;
    if (detTest < 0)
    {
        detTest = detTest * (-1);
    }

    // ray and triangle are parallel if det is close to 0
    if (detTest < kEpsilon) return false;

    float invDet = 1 / det;

    float tvec[3];
    customSubtract(orig, v0, tvec);

    float tempResult;
    customDotProduct(tvec, pvec, tempResult);
    u = tempResult * invDet;

    if (u < 0 || u > 1) return false;

    float qvec[3];
    customCrossProduct(tvec, v0v1, qvec);

    float tempResult1;
    customDotProduct(dir, qvec, tempResult1);
    v = tempResult1 * invDet;

    if (v < 0 || u + v > 1) return false;

    float tempResult3;
    customDotProduct(v0v2, qvec, tempResult3);
    t = tempResult3 * invDet;

    return true;
}

void getSurfaceProperties(
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float texCoordinates[NUM_TRIS * 3][2],
    const uint32_t &triIndex,
    float uv[2],
    float hitNormal[3],
    float hitTextureCoordinates[2])
{
    // face normal
    float v[3][3];
    float v0[3], v1[3], v2[3];
    for (int i = 0; i < 3; ++i)
    {
        copy3(P[trisIndex[triIndex*3 + i]], v[i]);
    }

    float subv1v0[3], subv2v0[3];
    customSubtract(v[1], v[0], subv1v0);
    customSubtract(v[2], v[0], subv2v0);
    customCrossProduct(subv1v0, subv2v0, hitNormal);
    customNormalize3(hitNormal);

    // texture coordinates
    float st0[2], st1[2], st2[3];
    copy2(texCoordinates[triIndex * 3], st0);
    copy2(texCoordinates[triIndex * 3 + 1], st1);
    copy2(texCoordinates[triIndex * 3 + 2], st2);
    // TODO: Check this
    for (int i = 0; i < 2; ++i)
    {
        hitTextureCoordinates[i] = (1 - uv[0] - uv[1]) * st0[i] + uv[0] * st1[i] + uv[1] * st2[i];
    }
}

void getPrimitive(
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float v0Arr[3], float v1Arr[3], float v2Arr[3],
    uint32_t index)
{
    uint32_t j = index*3;

    for (int i = 0; i < 3; ++i)
    {
        v0Arr[i] = P[trisIndex[j]][i];
        v1Arr[i] = P[trisIndex[j + 1]][i];
        v2Arr[i] = P[trisIndex[j + 2]][i];
    }
}

// Test if the ray interesests this triangle mesh
bool intersect(
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float origArr[3], float dirArr[3],
    float &tNear, uint32_t &triIndex,
    float uv[2])
{
    bool isect = false;
    for (uint32_t i = 0; i < NUM_TRIS; ++i) {
        float t = kInfinity, u, v;
        float v0Arr[3], v1Arr[3], v2Arr[3];
        getPrimitive(P, trisIndex, v0Arr, v1Arr, v2Arr, i);
        if (rayTriangleIntersect(origArr, dirArr, v0Arr, v1Arr, v2Arr, t, u, v) && t < tNear) {
            tNear = t;
            uv[0] = u;
            uv[1] = v;
            triIndex = i;
            isect = true;
        }
    }

    return isect;
}

bool trace(
    float orig[3], float dir[3],
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float &tNear, uint32_t &index, float uv[2])
{
    bool isIntersecting = false;
    float tNearTriangle = kInfinity;
    uint32_t indexTriangle;
    if (intersect(P, trisIndex, orig, dir, tNearTriangle, indexTriangle, uv) && tNearTriangle < tNear)
    {
        tNear = tNearTriangle;
        index = indexTriangle;

        isIntersecting = true;
    }

    return isIntersecting;
}

void castRay(
    float orig[3], float dir[3],
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float texCoordinates[NUM_TRIS * 3][2],
    float hitColor[3],
    float backgroundColor[3])
{
    for (int i = 0; i < 3; ++i)
    {
        hitColor[i] = backgroundColor[i];
    }

    float tnear = kInfinity;
    float uv[2];
    uint32_t index = 0;
    if (trace(orig, dir, P, trisIndex, tnear, index, uv))
    {
        float hitPoint[3];
        for (int i = 0; i < 3; ++i)
        {
            hitPoint[i] = orig[i] + dir[i] * tnear;
        }

        float hitNormal[2];
        float hitTexCoordinates[2];
        getSurfaceProperties(P, trisIndex, texCoordinates, index, uv, hitNormal, hitTexCoordinates);
        float neg_dir[3] = {-dir[0], -dir[1], -dir[2]};
        float normal_dir_dot;
        customDotProduct(hitNormal, neg_dir, normal_dir_dot);
        float NdotView = std::max(0.f, normal_dir_dot);
        const int M = 4;
        float checker = (fmod(hitTexCoordinates[0] * M, 1.0) > 0.5) ^ (fmod(hitTexCoordinates[1] * M, 1.0) < 0.5);
        float c = 0.3 * (1 - checker) + 0.7 * checker;

        for (int i = 0; i < 3; ++i)
        {
            hitColor[i] = c * NdotView;
        }
    }
}

// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
void render(
    float P[MAX_VERT_INDEX][3],
    uint32_t trisIndex[NUM_TRIS * 3],
    float texCoordinates[NUM_TRIS * 3][2],
    float framebuffer[WIDTH * HEIGHT][3],
    float cameraToWorld[4][4],
    float backgroundColor[3])
{
    float scale = tan(customDeg2Rad(FOV * 0.5));
    float imageAspectRatio = WIDTH / (float)HEIGHT;
    float origArr[3];
    float zeroArr[3] = {0, 0, 0};
    customMultVecMatrix(zeroArr, origArr, cameraToWorld);

    for (uint32_t j = 0; j < HEIGHT;  ++j) // HEIGHT;
    {
        for (uint32_t i = 0; i < WIDTH; ++i)
        {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)WIDTH - 1) * imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)HEIGHT) * scale;

            float srcRayDir[3] = {x, y, -1};
            float dirArr[3];

            customMultDirMatrix(srcRayDir, dirArr, cameraToWorld);

            customNormalize3(dirArr);
            castRay(origArr, dirArr, P, trisIndex, texCoordinates, &framebuffer[j*WIDTH + i][0], backgroundColor);
        }
        fprintf(stderr, "\r%3d%c", uint32_t(j / (float)HEIGHT * 100), '%');
    }
}