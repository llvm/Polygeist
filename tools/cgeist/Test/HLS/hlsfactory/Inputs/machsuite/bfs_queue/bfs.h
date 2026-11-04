/*
Implementation based on:
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "support.h"

// Terminology (but not values) from graph500 spec
//   graph density = 2^-(2*SCALE - EDGE_FACTOR)
#define SCALE 8
#define EDGE_FACTOR 16

#define N_NODES (1<<SCALE)
#define N_EDGES (N_NODES*EDGE_FACTOR)

// upper limit
#define N_LEVELS 10

// Larger than necessary for small graphs, but appropriate for large ones
typedef uint64_t edge_index_t;
typedef uint64_t node_index_t;

typedef struct edge_t_struct {
  // These fields are common in practice, but we elect not to use them.
  //weight_t weight;
  //node_index_t src;
  node_index_t dst;
} edge_t;

typedef struct node_t_struct {
  edge_index_t edge_begin;
  edge_index_t edge_end;
} node_t;

typedef int8_t level_t;
#define MAX_LEVEL INT8_MAX

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  node_t nodes[N_NODES];
  edge_t edges[N_EDGES];
  node_index_t starting_node;
  level_t level[N_NODES];
  edge_index_t level_counts[N_LEVELS];
};

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES], node_index_t starting_node, level_t level[N_NODES], edge_index_t level_counts[N_LEVELS]);
