// Minimal MLIR runner utility support needed by residual memref.copy lowering.
// This intentionally mirrors MLIR's CRunnerUtils memrefCopy ABI, but stays C
// so it can be linked by the host and Jetson cross build paths.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int64_t rank;
  void *descriptor;
} PolygeistUnrankedMemRef;

typedef struct {
  char *allocated;
  char *aligned;
  int64_t offset;
  int64_t sizesAndStrides[];
} PolygeistRankedMemRef;

void memrefCopy(int64_t elemSize, PolygeistUnrankedMemRef *srcArg,
                PolygeistUnrankedMemRef *dstArg) {
  int64_t rank = srcArg->rank;
  if (rank < 0)
    abort();
  PolygeistRankedMemRef *src = (PolygeistRankedMemRef *)srcArg->descriptor;
  PolygeistRankedMemRef *dst = (PolygeistRankedMemRef *)dstArg->descriptor;
  int64_t *srcSizes = src->sizesAndStrides;
  int64_t *srcStrides = src->sizesAndStrides + rank;
  int64_t *dstSizes = dst->sizesAndStrides;
  int64_t *dstStrides = dst->sizesAndStrides + rank;

  for (int64_t i = 0; i < rank; ++i)
    if (srcSizes[i] == 0)
      return;

  char *srcPtr = src->aligned + src->offset * elemSize;
  char *dstPtr = dst->aligned + dst->offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, (size_t)elemSize);
    return;
  }

  int64_t *indices = (int64_t *)calloc((size_t)rank, sizeof(int64_t));
  int64_t *srcByteStrides = (int64_t *)malloc((size_t)rank * sizeof(int64_t));
  int64_t *dstByteStrides = (int64_t *)malloc((size_t)rank * sizeof(int64_t));
  if (!indices || !srcByteStrides || !dstByteStrides)
    abort();

  for (int64_t i = 0; i < rank; ++i) {
    srcByteStrides[i] = srcStrides[i] * elemSize;
    dstByteStrides[i] = dstStrides[i] * elemSize;
  }

  int64_t readIndex = 0;
  int64_t writeIndex = 0;
  for (;;) {
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, (size_t)elemSize);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t next = ++indices[axis];
      readIndex += srcByteStrides[axis];
      writeIndex += dstByteStrides[axis];
      if (next != srcSizes[axis])
        break;
      if (axis == 0) {
        free(indices);
        free(srcByteStrides);
        free(dstByteStrides);
        return;
      }
      indices[axis] = 0;
      readIndex -= srcSizes[axis] * srcByteStrides[axis];
      writeIndex -= dstSizes[axis] * dstByteStrides[axis];
    }
  }
}
