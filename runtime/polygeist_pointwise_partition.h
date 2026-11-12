#ifndef POLYGEIST_POINTWISE_PARTITION_H
#define POLYGEIST_POINTWISE_PARTITION_H
#include <stdint.h>
#include <string.h>

/* Graph value numbers: inputs 0..3, scalar slots 4..11, instructions 12..35.
 * A cut is a materialized FP32 multiplication result. Boolean values stay
 * inside a stage, so the existing typed pointwise interface is sufficient. */
struct pg_pointwise_stage {
  uint64_t words[12];
  int nodes, external_count, external[4], output;
};
struct pg_pointwise_partition {
  int count;
  uint32_t cuts;
  unsigned used_inputs;
  struct pg_pointwise_stage stages[24];
};
struct pg_pointwise_builder {
  const uint64_t *words;
  int (*arity)(int);
  uint32_t produced;
  int mapped[24];
  struct pg_pointwise_stage *stage;
};
static uint32_t pg_pointwise_instruction(const uint64_t words[12], int i) {
  return (uint32_t)(words[i / 2] >> (32 * (i % 2)));
}
static int pg_pointwise_external(struct pg_pointwise_stage *s, int ref) {
  for (int i = 0; i < s->external_count; ++i)
    if (s->external[i] == ref) return i;
  if (s->external_count == 4) return -1;
  s->external[s->external_count] = ref;
  return s->external_count++;
}
static int pg_pointwise_emit(struct pg_pointwise_builder *b, int ref) {
  if (ref >= 4 && ref < 12) return ref;
  if (ref < 4 || (b->produced & (1u << (ref - 12))))
    return pg_pointwise_external(b->stage, ref);
  int node = ref - 12;
  if (b->mapped[node] >= 0) return b->mapped[node];
  uint32_t inst = pg_pointwise_instruction(b->words, node);
  int opcode = (int)(inst >> 24), arity = b->arity(opcode);
  uint32_t result = (uint32_t)opcode << 24;
  for (int k = 0; k < arity; ++k) {
    int operand = (int)((inst >> (16 - 8 * k)) & 255);
    int translated = pg_pointwise_emit(b, operand);
    if (translated < 0) return -1;
    result |= (uint32_t)translated << (16 - 8 * k);
  }
  int i = b->stage->nodes++;
  if (i >= 24) return -1;
  b->stage->words[i / 2] |= (uint64_t)result << (32 * (i % 2));
  return b->mapped[node] = 12 + i;
}

/* Return 0 when fusion cannot contract a multiplication into ADD/SUB, 1 for
 * a valid partition, or -1 when the existing four-input stage ABI cannot
 * represent the required cuts. Invalid references/opcodes are rejected. */
static int pg_pointwise_partition_build_mixed(
    const uint64_t words[12], int nodes, int (*arity)(int),
    int (*is_multiply)(int), int (*is_add_subtract)(int),
    int (*requires_separate_stage)(int), int (*is_boolean)(int),
    struct pg_pointwise_partition *out) {
  memset(out, 0, sizeof(*out));
  if (nodes <= 0 || nodes > 24) return -1;
  uint32_t ancestors[24] = {0}, multiplies[24] = {0};
  unsigned input_masks[24] = {0};
  uint32_t needed_cuts = 0;
  for (int i = 0; i < nodes; ++i) {
    uint32_t inst = pg_pointwise_instruction(words, i);
    int opcode = (int)(inst >> 24), count = arity(opcode);
    if (count < 1 || count > 3) return -1;
    for (int k = 0; k < count; ++k) {
      int ref = (int)((inst >> (16 - 8 * k)) & 255);
      if (ref < 4) input_masks[i] |= 1u << ref;
      else if (ref >= 12) {
        int producer = ref - 12;
        if (producer >= i) return -1;
        ancestors[i] |= ancestors[producer] | (1u << producer);
        multiplies[i] |= multiplies[producer];
        input_masks[i] |= input_masks[producer];
      }
    }
    if (is_multiply(opcode)) multiplies[i] |= 1u << i;
  }
  uint32_t live = ancestors[nodes - 1] | (1u << (nodes - 1));
  /* An unrelated dead ADD must not force cuts of a live multiplication. */
  needed_cuts = 0;
  for (int i = 0; i < nodes; ++i)
    if ((live & (1u << i)) &&
        is_add_subtract((int)(pg_pointwise_instruction(words, i) >> 24)))
      needed_cuts |= multiplies[i];
  /* An operator assigned to another typed backend must form a standalone
   * stage. Materialize its expression operands first; preserve shared DAG
   * identities and never pass a Boolean intermediate through an FP32 array. */
  if (requires_separate_stage) {
    for (int i = 0; i < nodes; ++i) {
      uint32_t inst = pg_pointwise_instruction(words, i);
      int opcode = (int)(inst >> 24);
      if (!(live & (1u << i)) || !requires_separate_stage(opcode)) continue;
      needed_cuts |= 1u << i;
      for (int k = 0; k < arity(opcode); ++k) {
        int ref = (int)((inst >> (16 - 8 * k)) & 255);
        if (ref >= 12) needed_cuts |= 1u << (ref - 12);
      }
    }
  }
  if (!needed_cuts) return 0;
  if (is_boolean)
    for (int i = 0; i < nodes; ++i)
      if ((needed_cuts & (1u << i)) &&
          is_boolean((int)(pg_pointwise_instruction(words, i) >> 24))) return -1;
  out->cuts = needed_cuts;
  out->used_inputs = input_masks[nodes - 1];
  uint32_t produced = 0;
  for (int i = 0; i < nodes; ++i) {
    if (!(needed_cuts & (1u << i)) && i != nodes - 1) continue;
    struct pg_pointwise_stage *stage = &out->stages[out->count++];
    struct pg_pointwise_builder builder = {words, arity, produced, {0}, stage};
    for (int j = 0; j < 24; ++j) builder.mapped[j] = -1;
    stage->output = i;
    if (pg_pointwise_emit(&builder, 12 + i) < 0) return -1;
    produced |= 1u << i;
  }
  return 1;
}
/* Original pure-cuDNN/FMA planning remains available to structural callers. */
static inline int pg_pointwise_partition_build(
    const uint64_t words[12], int nodes, int (*arity)(int),
    int (*is_multiply)(int), int (*is_add_subtract)(int),
    struct pg_pointwise_partition *out) {
  return pg_pointwise_partition_build_mixed(words, nodes, arity, is_multiply,
                                           is_add_subtract, NULL, NULL, out);
}
#endif
