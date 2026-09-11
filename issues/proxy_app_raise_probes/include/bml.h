#ifndef PROXY_APP_RAISE_PROBE_BML_H
#define PROXY_APP_RAISE_PROBE_BML_H

typedef double real_t;
typedef struct bml_matrix_t bml_matrix_t;

real_t *bml_gershgorin(bml_matrix_t *matrix);
void bml_scale_add_identity(bml_matrix_t *matrix, real_t alpha, real_t beta,
                            real_t threshold);
void bml_free_memory(void *ptr);
void bml_copy(const bml_matrix_t *from, bml_matrix_t *to);
bml_matrix_t *bml_copy_new(const bml_matrix_t *matrix);
real_t bml_trace(const bml_matrix_t *matrix);
real_t *bml_multiply_x2(const bml_matrix_t *matrix, bml_matrix_t *x2,
                        real_t threshold);
void bml_add(bml_matrix_t *x, bml_matrix_t *y, real_t alpha, real_t beta,
             real_t threshold);
int bml_printRank(void);
int bml_getNRanks(void);
int bml_get_bandwidth(const bml_matrix_t *matrix);
void bml_scale_inplace(const real_t *scale, bml_matrix_t *matrix);
void bml_deallocate(bml_matrix_t **matrix);

#endif
