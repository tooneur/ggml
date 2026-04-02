#ifndef MODULAR_ENGINE_PARSER_H
#define MODULAR_ENGINE_PARSER_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "ggml.h"
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

struct model {
    char *model_name;
    struct ggml_context *ctx;
    struct gguf_context *ctxgguf;
    int n_tensors;
    struct {
        int n_layer;
        int n_ctx;
        int n_embd;
        int n_ff;
        int n_head;
        int n_vocab;
    } hparams;
};

bool model_load(const char *filename, struct model *model);
bool run_single_token_inference(struct model *model, int token_id);
bool run_batch_inference(struct model *model, const int *token_ids, int n_tokens, int n_past);
void save_logits(float *logits, int n_vocab);
static struct ggml_tensor *find_tensor_any(
    struct ggml_context *ctx,
    const char *const *names,
    int n_names,
    const char **matched_name);
static bool load_required_tensor_any(
    struct ggml_context *ctx,
    struct ggml_tensor **out,
    const char *const *names,
    int n_names,
    const char *label);

#ifdef __cplusplus
}
#endif

#endif

