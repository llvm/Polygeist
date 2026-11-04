/*
Based on:
Lawrence Rabiner. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." Proc. IEEE, v77, #2. 1989.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include "support.h"

#define TYPE double
typedef uint8_t tok_t;
typedef TYPE prob_t;
typedef uint8_t state_t;
typedef int32_t step_t;

//#define N_STATES 5
//#define N_OBS 32
//#define N_TOKENS 9
#define N_STATES  64
#define N_OBS     140
#define N_TOKENS  64

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] );

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  tok_t obs[N_OBS];
  prob_t init[N_STATES];
  prob_t transition[N_STATES*N_STATES];
  prob_t emission[N_STATES*N_TOKENS];
  state_t path[N_OBS];
};
