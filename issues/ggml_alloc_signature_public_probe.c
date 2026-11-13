#include "ggml-alloc.h"

enum ggml_status isolate_tallocr_public(
        struct ggml_tallocr * talloc,
        struct ggml_tensor * tensor) {
    (void)talloc;
    (void)tensor;
    return (enum ggml_status)0;
}

bool isolate_gallocr_public(
        ggml_gallocr_t galloc,
        struct ggml_cgraph * graph,
        const int * node_buffer_ids,
        const int * leaf_buffer_ids) {
    (void)galloc;
    (void)graph;
    (void)node_buffer_ids;
    (void)leaf_buffer_ids;
    return false;
}

ggml_backend_buffer_t isolate_backend_alloc_public(
        struct ggml_context * ctx,
        ggml_backend_buffer_type_t buft) {
    (void)ctx;
    (void)buft;
    return (ggml_backend_buffer_t)0;
}
