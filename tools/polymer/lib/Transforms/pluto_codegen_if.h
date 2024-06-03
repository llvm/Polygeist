#ifndef _PLUTO_CODEGEN_IF_H_
#define _PLUTO_CODEGEN_IF_H_

#include "osl/extensions/loop.h"

typedef struct plutoProg PlutoProg;

osl_loop_p pluto_get_vector_loop_list(const PlutoProg *prog);
osl_loop_p pluto_get_parallel_loop_list(const PlutoProg *prog, int vloopsfound);
#endif
