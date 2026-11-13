typedef struct {
  char *alloc_buffer;
} stb_vorbis_alloc;

typedef struct {
  stb_vorbis_alloc alloc;
  int temp_offset;
} vorb;

void *setup_temp_malloc(vorb *f, int sz);

void inverse_mdct_ptrloop(float *buffer, int n, vorb *f, int blocktype) {
  int n2 = n >> 1;
  float *buf2 = (float *)(f->alloc.alloc_buffer
                              ? setup_temp_malloc(f, n2 * sizeof(*buf2))
                              : __builtin_alloca(n2 * sizeof(*buf2)));
  float *d = &buf2[n2 - 2];
  (void)buffer;
  (void)blocktype;
  while (d >= buf2) {
    d[0] = 0.0f;
    d -= 2;
  }
}
