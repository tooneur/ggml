#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "parser.h"
#include "distrib.h"



static struct ggml_tensor *find_tensor_any(
    struct ggml_context *ctx,
    const char *const *names,
    int n_names,
    const char **matched_name)
{

    for (int i = 0; i < n_names; ++i)
    {
        struct ggml_tensor *t = ggml_get_tensor(ctx, names[i]);
        if (t != NULL)
        {
            if (matched_name != NULL)
            {
                *matched_name = names[i];
            }
            return t;
        }
    }

    if (matched_name != NULL)
    {
        *matched_name = NULL;
    }
    return NULL;
}

void save_logits(float *logits, int n_vocab)
{
    FILE *f = fopen("logits-custom.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);

    printf("Logits saved to logits-custom.bin\n");
}

static bool can_mul_mat_shapes(const struct ggml_tensor *a, const struct ggml_tensor *b)
{
    return (a->ne[0] == b->ne[0]) &&
           (b->ne[2] % a->ne[2] == 0) &&
           (b->ne[3] % a->ne[3] == 0);
}

static struct ggml_tensor *mul_mat(
    struct ggml_context *ctx,
    struct ggml_tensor *w,
    struct ggml_tensor *x,
    const char *label)
{
    if (w == NULL || x == NULL) {
        printf("mul_mat: NULL tensor for %s\n", label);
        return NULL;
    }

    // Vérifie si les dimensions principales sont compatibles pour ggml_mul_mat
    // ggml_mul_mat(a, b) attend a->ne[0] == b->ne[0]
    if (w->ne[0] != x->ne[0]) {
        printf("mul_mat shape mismatch for %s\n", label);
        printf("  w shape: [%lld, %lld, %lld, %lld]\n",
               (long long)w->ne[0], (long long)w->ne[1],
               (long long)w->ne[2], (long long)w->ne[3]);
        printf("  x shape: [%lld, %lld, %lld, %lld]\n",
               (long long)x->ne[0], (long long)x->ne[1],
               (long long)x->ne[2], (long long)x->ne[3]);
        return NULL;
    }

    // Tout est OK → juste multiplier
    return ggml_mul_mat(ctx, w, x);
}

static bool load_required_tensor_any(
    struct ggml_context *ctx,
    struct ggml_tensor **out,
    const char *const *names,
    int n_names,
    const char *label)
{
    const char *matched = NULL;
    *out = find_tensor_any(ctx, names, n_names, &matched);
    if (*out == NULL)
    {
        printf("Tensor not found for %s. Tried:\n", label);
        for (int i = 0; i < n_names; ++i)
        {
            printf("  - %s\n", names[i]);
        }
        return false;
    }

    return true;
}

static char *gguf_get_model_name(const char *path, struct gguf_context *ctxgguf)
{
    if (ctxgguf == NULL)
    {
        fprintf(stderr, "Failed to initialize gguf context from file: %s\n", path);
        return NULL;
    }

    int key = gguf_find_key(ctxgguf, "general.architecture");
    if (key < 0)
    {
        fprintf(stderr, "Key 'general.architecture' not found\n");
        return NULL;
    }

    const char *model_name = gguf_get_val_str(ctxgguf, key);
    printf("Model name: %s\n", model_name);

    return strdup(model_name);
}

static int gguf_find_key_with_custom_model_name(struct gguf_context *ctxgguf, const char *key, const char *model_name)
{

    int result = -1;
    char new_key[256];

    if (key[0] == '.')
    {
        snprintf(new_key, sizeof(new_key), "%s%s", model_name, key);
    }
    else
    {
        snprintf(new_key, sizeof(new_key), "%s.%s", model_name, key);
    }
    result = gguf_find_key(ctxgguf, new_key);
    if (result >= 0)
    {
        printf("Found key '%s' for model '%s'\n", new_key, model_name);
        return result;
    }
    return result;
}

static int gguf_find_first_key(struct gguf_context *ctxgguf, const char *model_name, const char *const *keys, int n_keys)
{
    for (int i = 0; i < n_keys; ++i)
    {
        int key = -1;
        if (model_name != NULL && model_name[0] != '\0')
        {
            key = gguf_find_key_with_custom_model_name(ctxgguf, keys[i], model_name);
        }
        if (key < 0)
        {
            key = gguf_find_key(ctxgguf, keys[i]);
        }
        if (key >= 0)
        {
            return key;
        }
    }

    return -1;
}

static bool gguf_read_u32_with_fallback(
    struct gguf_context *ctxgguf,
    const char *model_name,
    const char *field_name,
    const char *const *keys,
    int n_keys,
    int *out)
{
    int key = gguf_find_first_key(ctxgguf, model_name, keys, n_keys);
    if (key < 0)
    {
        printf("Key not found for %s. Tried:\n", field_name);
        for (int i = 0; i < n_keys; ++i)
        {
            printf("  - %s.%s\n", model_name != NULL ? model_name : "<arch>", keys[i]);
            printf("  - %s\n", keys[i]);
        }
        return false;
    }

    *out = (int)gguf_get_val_u32(ctxgguf, key);
    return true;
}

static char *gguf_tensor_name_reader(struct gguf_context *ctx, int index, struct gguf_context *ctxgguf)
{

    int n_tensors = gguf_get_n_tensors(ctxgguf);
    printf("Number of tensors in model: %d\n", n_tensors);

    char *result = NULL;
    char *name = NULL;

    for (int i = 0; i < n_tensors; i++)
    {
        const char *name = gguf_get_tensor_name(ctxgguf, i);
        printf("Tensor %d: %s\n", i, name);
        result = strdup(name);
        break;
    }

    return result;
}

static int gguf_hyperparameter_reader(char *result, struct gguf_context *ctxgguf, struct model *model)
{

    // lecture des hyperparametre
    int key;
    for (int i = 0; i < gguf_get_n_kv(ctxgguf); i++)
    {
        const char *name = gguf_get_key(ctxgguf, i);
        printf("Meta %d: %s\n", i, name);
    }

    const char *N_LAYER_KEYS[] = {"block_count", "n_layer"};
    const char *N_CTX_KEYS[] = {"context_length", "n_ctx"};
    const char *N_EMBD_KEYS[] = {"embedding_length", "n_embd"};
    const char *N_FF_KEYS[] = {"feed_forward_length", "n_ff"};
    const char *N_HEAD_KEYS[] = {"attention.head_count", "n_head"};

    if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_layer", N_LAYER_KEYS, 2, &model->hparams.n_layer))
    {
        return false;
    }

    if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_ctx", N_CTX_KEYS, 2, &model->hparams.n_ctx))
    {
        return false;
    }

    if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_embd", N_EMBD_KEYS, 2, &model->hparams.n_embd))
    {
        return false;
    }

    if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_ff", N_FF_KEYS, 2, &model->hparams.n_ff))
    {
        return false;
    }

    if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_head", N_HEAD_KEYS, 2, &model->hparams.n_head))
    {
        return false;
    }

    // n_vocab
    const char *N_VOCAB_ARR_KEYS[] = {"tokenizer.ggml.tokens"};
    key = gguf_find_first_key(ctxgguf, model->model_name, N_VOCAB_ARR_KEYS, 1);
    if (key < 0)
    {
        const char *N_VOCAB_KEYS[] = {"vocab_size", "tokenizer.ggml.vocab_size"};
        if (!gguf_read_u32_with_fallback(ctxgguf, model->model_name, "n_vocab", N_VOCAB_KEYS, 2, &model->hparams.n_vocab))
        {
            return false;
        }
    }
    else
    {
        model->hparams.n_vocab = gguf_get_arr_n(ctxgguf, key);
    }

    return true;
}

bool gguf_tensor_loader(struct ggml_context *ctx, struct gguf_context *ctxgguf, struct model *model, const char *const *names, int n_names, const char *label, const int n_tensors)
{

    // int n_tensors = gguf_get_n_tensors(ctxgguf);
    printf("Number of tensors in model: %d\n", n_tensors);

    for (int i = 0; i < n_tensors; i++)
    {
        const char *name = gguf_get_tensor_name(ctxgguf, i);
        printf("Tensor %d: %s\n", i, name);
    }

    const char *matched_name = NULL;
    struct ggml_tensor *tensor = find_tensor_any(ctx, names, n_names, &matched_name);
    if (tensor == NULL)
    {
        printf("Tensor not found for %s. Tried:\n", label);
        for (int i = 0; i < n_names; ++i)
        {
            printf("  - %s\n", names[i]);
        }
        return false;
    }

    printf("Loaded tensor '%s' for %s\n", matched_name, label);
    return true;
}

bool model_load(const char *filename, struct model *model)
{

    struct ggml_context *ctx = NULL;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx,
    };

    struct gguf_context *ctxgguf = gguf_init_from_file(filename, params);

    if (ctxgguf == NULL)
    {
        fprintf(stderr, "Failed to initialize gguf context from file: %s\n", filename);
        return false;
    }

    model->model_name = gguf_get_model_name(filename, ctxgguf);

    model->ctx = ctx;
    model->ctxgguf = ctxgguf;

    model->n_tensors = gguf_get_n_tensors(ctxgguf);

    for (int i = 0; i < model->n_tensors; i++)
    {
        const char *name = gguf_get_tensor_name(ctxgguf, i);
        printf("Tensor %d: %s\n", i, name);
    }

    if (!gguf_hyperparameter_reader(NULL, ctxgguf, model))
    {
        fprintf(stderr, "Failed to read hyperparameters from model file: %s\n", filename);
        gguf_free(ctxgguf);
        return false;
    }

    return true;
}

struct inference_global_tensors
{
    struct ggml_tensor *wte;
    struct ggml_tensor *wpe;
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;
    struct ggml_tensor *lm_head;
};

struct inference_layer_tensors
{
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;

    // Fused QKV path (GPT-2 style)
    struct ggml_tensor *c_attn_w;
    struct ggml_tensor *c_attn_b;

    // Split Q/K/V path (OPT-like)
    struct ggml_tensor *q_attn_w;
    struct ggml_tensor *q_attn_b;
    struct ggml_tensor *k_attn_w;
    struct ggml_tensor *k_attn_b;
    struct ggml_tensor *v_attn_w;
    struct ggml_tensor *v_attn_b;

    struct ggml_tensor *c_proj_w;
    struct ggml_tensor *c_proj_b;

    struct ggml_tensor *ln_2_g;
    struct ggml_tensor *ln_2_b;

    struct ggml_tensor *c_fc_w;
    struct ggml_tensor *c_fc_b;
    struct ggml_tensor *c_proj2_w;
    struct ggml_tensor *c_proj2_b;
};

struct inference_name_list
{
    const char *const *names;
    int n_names;
    const char *label;
    bool required;
};

struct inference_pattern_list
{
    const char *const *patterns;
    int n_patterns;
    const char *label;
    bool required;
};

struct inference_profile
{
    const char *profile_name;
    bool qkv_fused;

    struct inference_name_list wte;
    struct inference_name_list wpe;
    struct inference_name_list ln_f_g;
    struct inference_name_list ln_f_b;
    struct inference_name_list lm_head;

    struct inference_pattern_list ln_1_g;
    struct inference_pattern_list ln_1_b;

    struct inference_pattern_list c_attn_w;
    struct inference_pattern_list c_attn_b;
    struct inference_pattern_list q_attn_w;
    struct inference_pattern_list q_attn_b;
    struct inference_pattern_list k_attn_w;
    struct inference_pattern_list k_attn_b;
    struct inference_pattern_list v_attn_w;
    struct inference_pattern_list v_attn_b;

    struct inference_pattern_list c_proj_w;
    struct inference_pattern_list c_proj_b;
    struct inference_pattern_list ln_2_g;
    struct inference_pattern_list ln_2_b;
    struct inference_pattern_list c_fc_w;
    struct inference_pattern_list c_fc_b;
    struct inference_pattern_list c_proj2_w;
    struct inference_pattern_list c_proj2_b;
};

static const char *NAMES_WTE[] = {
    "token_embd.weight",
    "transformer.wte.weight",
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
};

static const char *NAMES_WPE[] = {
    "position_embd.weight",
    "transformer.wpe.weight",
    "model.decoder.embed_positions.weight",
};

static const char *NAMES_LN_F_G[] = {
    "output_norm.weight",
    "transformer.ln_f.weight",
    "model.decoder.final_layer_norm.weight",
    "model.norm.weight",
    "norm.weight",
};

static const char *NAMES_LN_F_B[] = {
    "output_norm.bias",
    "transformer.ln_f.bias",
    "model.decoder.final_layer_norm.bias",
    "model.norm.bias",
    "norm.bias",
};

static const char *NAMES_LM_HEAD[] = {
    "output.weight",
    "lm_head.weight",
    "model.decoder.output_projection.weight",
};

static const char *PATS_LN_1_G[] = {
    "blk.%d.attn_norm.weight",
    "transformer.h.%d.ln_1.weight",
    "model.decoder.layers.%d.self_attn_layer_norm.weight",
};

static const char *PATS_LN_1_B[] = {
    "blk.%d.attn_norm.bias",
    "transformer.h.%d.ln_1.bias",
    "model.decoder.layers.%d.self_attn_layer_norm.bias",
};

static const char *PATS_C_ATTN_W[] = {
    "blk.%d.attn_qkv.weight",
    "transformer.h.%d.attn.c_attn.weight",
};

static const char *PATS_C_ATTN_B[] = {
    "blk.%d.attn_qkv.bias",
    "transformer.h.%d.attn.c_attn.bias",
    "transformer.h.%d.attn.c_attn.b",
};

static const char *PATS_Q_ATTN_W[] = {
    "model.decoder.layers.%d.self_attn.q_proj.weight",
};

static const char *PATS_Q_ATTN_B[] = {
    "model.decoder.layers.%d.self_attn.q_proj.bias",
};

static const char *PATS_K_ATTN_W[] = {
    "model.decoder.layers.%d.self_attn.k_proj.weight",
};

static const char *PATS_K_ATTN_B[] = {
    "model.decoder.layers.%d.self_attn.k_proj.bias",
};

static const char *PATS_V_ATTN_W[] = {
    "model.decoder.layers.%d.self_attn.v_proj.weight",
};

static const char *PATS_V_ATTN_B[] = {
    "model.decoder.layers.%d.self_attn.v_proj.bias",
};

static const char *PATS_C_PROJ_W[] = {
    "blk.%d.attn_output.weight",
    "transformer.h.%d.attn.c_proj.weight",
    "model.decoder.layers.%d.self_attn.out_proj.weight",
};

static const char *PATS_C_PROJ_B[] = {
    "blk.%d.attn_output.bias",
    "transformer.h.%d.attn.c_proj.bias",
    "transformer.h.%d.attn.c_proj.b",
    "model.decoder.layers.%d.self_attn.out_proj.bias",
};

static const char *PATS_LN_2_G[] = {
    "blk.%d.ffn_norm.weight",
    "transformer.h.%d.ln_2.weight",
    "model.decoder.layers.%d.final_layer_norm.weight",
};

static const char *PATS_LN_2_B[] = {
    "blk.%d.ffn_norm.bias",
    "transformer.h.%d.ln_2.bias",
    "transformer.h.%d.ln_2.b",
    "model.decoder.layers.%d.final_layer_norm.bias",
};

static const char *PATS_C_FC_W[] = {
    "blk.%d.ffn_up.weight",
    "transformer.h.%d.mlp.c_fc.weight",
    "model.decoder.layers.%d.fc1.weight",
};

static const char *PATS_C_FC_B[] = {
    "blk.%d.ffn_up.bias",
    "transformer.h.%d.mlp.c_fc.bias",
    "transformer.h.%d.mlp.c_fc.b",
    "model.decoder.layers.%d.fc1.bias",
};

static const char *PATS_C_PROJ2_W[] = {
    "blk.%d.ffn_down.weight",
    "transformer.h.%d.mlp.c_proj.weight",
    "model.decoder.layers.%d.fc2.weight",
};

static const char *PATS_C_PROJ2_B[] = {
    "blk.%d.ffn_down.bias",
    "transformer.h.%d.mlp.c_proj.bias",
    "transformer.h.%d.mlp.c_proj.b",
    "model.decoder.layers.%d.fc2.bias",
};

static const struct inference_profile PROFILE_DECODER_QKV_FUSED = {
    .profile_name = "decoder-qkv-fused",
    .qkv_fused = true,

    .wte = {.names = NAMES_WTE, .n_names = 4, .label = "wte", .required = true},
    .wpe = {.names = NAMES_WPE, .n_names = 3, .label = "wpe", .required = false},
    .ln_f_g = {.names = NAMES_LN_F_G, .n_names = 5, .label = "ln_f.weight", .required = true},
    .ln_f_b = {.names = NAMES_LN_F_B, .n_names = 5, .label = "ln_f.bias", .required = false},
    .lm_head = {.names = NAMES_LM_HEAD, .n_names = 3, .label = "lm_head", .required = true},

    .ln_1_g = {.patterns = PATS_LN_1_G, .n_patterns = 3, .label = "layer.ln_1_g", .required = true},
    .ln_1_b = {.patterns = PATS_LN_1_B, .n_patterns = 3, .label = "layer.ln_1_b", .required = false},
    .c_attn_w = {.patterns = PATS_C_ATTN_W, .n_patterns = 2, .label = "layer.c_attn_w", .required = true},
    .c_attn_b = {.patterns = PATS_C_ATTN_B, .n_patterns = 3, .label = "layer.c_attn_b", .required = false},
    .q_attn_w = {.patterns = NULL, .n_patterns = 0, .label = "layer.q_attn_w", .required = false},
    .q_attn_b = {.patterns = NULL, .n_patterns = 0, .label = "layer.q_attn_b", .required = false},
    .k_attn_w = {.patterns = NULL, .n_patterns = 0, .label = "layer.k_attn_w", .required = false},
    .k_attn_b = {.patterns = NULL, .n_patterns = 0, .label = "layer.k_attn_b", .required = false},
    .v_attn_w = {.patterns = NULL, .n_patterns = 0, .label = "layer.v_attn_w", .required = false},
    .v_attn_b = {.patterns = NULL, .n_patterns = 0, .label = "layer.v_attn_b", .required = false},
    .c_proj_w = {.patterns = PATS_C_PROJ_W, .n_patterns = 3, .label = "layer.c_proj_w", .required = true},
    .c_proj_b = {.patterns = PATS_C_PROJ_B, .n_patterns = 4, .label = "layer.c_proj_b", .required = false},
    .ln_2_g = {.patterns = PATS_LN_2_G, .n_patterns = 3, .label = "layer.ln_2_g", .required = true},
    .ln_2_b = {.patterns = PATS_LN_2_B, .n_patterns = 4, .label = "layer.ln_2_b", .required = false},
    .c_fc_w = {.patterns = PATS_C_FC_W, .n_patterns = 3, .label = "layer.c_fc_w", .required = true},
    .c_fc_b = {.patterns = PATS_C_FC_B, .n_patterns = 4, .label = "layer.c_fc_b", .required = false},
    .c_proj2_w = {.patterns = PATS_C_PROJ2_W, .n_patterns = 3, .label = "layer.c_proj2_w", .required = true},
    .c_proj2_b = {.patterns = PATS_C_PROJ2_B, .n_patterns = 4, .label = "layer.c_proj2_b", .required = false},
};

static const struct inference_profile PROFILE_DECODER_QKV_SPLIT = {
    .profile_name = "decoder-qkv-split",
    .qkv_fused = false,

    .wte = {.names = NAMES_WTE, .n_names = 4, .label = "wte", .required = true},
    .wpe = {.names = NAMES_WPE, .n_names = 3, .label = "wpe", .required = false},
    .ln_f_g = {.names = NAMES_LN_F_G, .n_names = 5, .label = "ln_f.weight", .required = true},
    .ln_f_b = {.names = NAMES_LN_F_B, .n_names = 5, .label = "ln_f.bias", .required = false},
    .lm_head = {.names = NAMES_LM_HEAD, .n_names = 3, .label = "lm_head", .required = true},

    .ln_1_g = {.patterns = PATS_LN_1_G, .n_patterns = 3, .label = "layer.ln_1_g", .required = true},
    .ln_1_b = {.patterns = PATS_LN_1_B, .n_patterns = 3, .label = "layer.ln_1_b", .required = false},
    .c_attn_w = {.patterns = NULL, .n_patterns = 0, .label = "layer.c_attn_w", .required = false},
    .c_attn_b = {.patterns = NULL, .n_patterns = 0, .label = "layer.c_attn_b", .required = false},
    .q_attn_w = {.patterns = PATS_Q_ATTN_W, .n_patterns = 1, .label = "layer.q_attn_w", .required = true},
    .q_attn_b = {.patterns = PATS_Q_ATTN_B, .n_patterns = 1, .label = "layer.q_attn_b", .required = false},
    .k_attn_w = {.patterns = PATS_K_ATTN_W, .n_patterns = 1, .label = "layer.k_attn_w", .required = true},
    .k_attn_b = {.patterns = PATS_K_ATTN_B, .n_patterns = 1, .label = "layer.k_attn_b", .required = false},
    .v_attn_w = {.patterns = PATS_V_ATTN_W, .n_patterns = 1, .label = "layer.v_attn_w", .required = true},
    .v_attn_b = {.patterns = PATS_V_ATTN_B, .n_patterns = 1, .label = "layer.v_attn_b", .required = false},
    .c_proj_w = {.patterns = PATS_C_PROJ_W, .n_patterns = 3, .label = "layer.c_proj_w", .required = true},
    .c_proj_b = {.patterns = PATS_C_PROJ_B, .n_patterns = 4, .label = "layer.c_proj_b", .required = false},
    .ln_2_g = {.patterns = PATS_LN_2_G, .n_patterns = 3, .label = "layer.ln_2_g", .required = true},
    .ln_2_b = {.patterns = PATS_LN_2_B, .n_patterns = 4, .label = "layer.ln_2_b", .required = false},
    .c_fc_w = {.patterns = PATS_C_FC_W, .n_patterns = 3, .label = "layer.c_fc_w", .required = true},
    .c_fc_b = {.patterns = PATS_C_FC_B, .n_patterns = 4, .label = "layer.c_fc_b", .required = false},
    .c_proj2_w = {.patterns = PATS_C_PROJ2_W, .n_patterns = 3, .label = "layer.c_proj2_w", .required = true},
    .c_proj2_b = {.patterns = PATS_C_PROJ2_B, .n_patterns = 4, .label = "layer.c_proj2_b", .required = false},
};

static bool profile_has_required_tensors(const struct model *model, const struct inference_profile *profile)
{
    struct ggml_tensor *tmp = NULL;
    const char *matched = NULL;

    tmp = find_tensor_any(model->ctx, profile->wte.names, profile->wte.n_names, &matched);
    if (tmp == NULL)
    {
        return false;
    }

    tmp = find_tensor_any(model->ctx, profile->lm_head.names, profile->lm_head.n_names, &matched);
    if (tmp == NULL)
    {
        return false;
    }

    char names_buf[6][128];
    const char *names[6];

    const struct inference_pattern_list *attn_req[3] = {0};
    int attn_req_count = 0;

    if (profile->qkv_fused)
    {
        attn_req[attn_req_count++] = &profile->c_attn_w;
    }
    else
    {
        attn_req[attn_req_count++] = &profile->q_attn_w;
        attn_req[attn_req_count++] = &profile->k_attn_w;
        attn_req[attn_req_count++] = &profile->v_attn_w;
    }

    for (int req = 0; req < attn_req_count; ++req)
    {
        const struct inference_pattern_list *spec = attn_req[req];
        if (spec->n_patterns <= 0 || spec->n_patterns > 6)
        {
            return false;
        }

        for (int i = 0; i < spec->n_patterns; ++i)
        {
            snprintf(names_buf[i], sizeof(names_buf[i]), spec->patterns[i], 0);
            names[i] = names_buf[i];
        }

        tmp = find_tensor_any(model->ctx, names, spec->n_patterns, &matched);
        if (tmp == NULL)
        {
            return false;
        }
    }

    return true;
}

static const struct inference_profile *get_inference_profile(const struct model *model)
{
    const struct inference_profile *candidates[] = {
        &PROFILE_DECODER_QKV_SPLIT,
        &PROFILE_DECODER_QKV_FUSED,
    };

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
    {
        if (profile_has_required_tensors(model, candidates[i]))
        {
            return candidates[i];
        }
    }

    // Keep legacy fallback behavior to avoid regressions on partially named checkpoints.
    return &PROFILE_DECODER_QKV_FUSED;
}

static bool load_tensor_from_name_list(
    struct ggml_context *ctx,
    struct ggml_tensor **out,
    const struct inference_name_list *spec)
{
    if (spec->required)
    {
        return load_required_tensor_any(ctx, out, spec->names, spec->n_names, spec->label);
    }

    const char *matched = NULL;
    *out = find_tensor_any(ctx, spec->names, spec->n_names, &matched);
    if (*out == NULL)
    {
        printf("Optional tensor missing for %s\n", spec->label);
    }
    return true;
}

static bool load_block_tensor_from_patterns(
    struct ggml_context *ctx,
    int layer_idx,
    struct ggml_tensor **out,
    const struct inference_pattern_list *spec)
{
    char names_buf[6][128];
    const char *names[6];

    if (spec->n_patterns <= 0 || spec->n_patterns > 6)
    {
        printf("Invalid pattern count for %s\n", spec->label);
        return false;
    }

    for (int i = 0; i < spec->n_patterns; ++i)
    {
        snprintf(names_buf[i], sizeof(names_buf[i]), spec->patterns[i], layer_idx);
        names[i] = names_buf[i];
    }

    if (spec->required)
    {
        return load_required_tensor_any(ctx, out, names, spec->n_patterns, spec->label);
    }

    const char *matched = NULL;
    *out = find_tensor_any(ctx, names, spec->n_patterns, &matched);
    if (*out == NULL)
    {
        printf("Optional tensor missing for %s at layer %d\n", spec->label, layer_idx);
    }

    return true;
}

static bool load_inference_global_tensors(
    struct model *model,
    const struct inference_profile *profile,
    struct inference_global_tensors *t)
{
    if (!load_tensor_from_name_list(model->ctx, &t->wte, &profile->wte))
    {
        return false;
    }
    if (!load_tensor_from_name_list(model->ctx, &t->wpe, &profile->wpe))
    {
        return false;
    }
    if (!load_tensor_from_name_list(model->ctx, &t->ln_f_g, &profile->ln_f_g))
    {
        return false;
    }
    if (!load_tensor_from_name_list(model->ctx, &t->ln_f_b, &profile->ln_f_b))
    {
        return false;
    }
    if (!load_tensor_from_name_list(model->ctx, &t->lm_head, &profile->lm_head))
    {
        return false;
    }

    return true;
}

static bool load_inference_layer_tensors(
    struct model *model,
    const struct inference_profile *profile,
    int layer_idx,
    struct inference_layer_tensors *t)
{
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->ln_1_g, &profile->ln_1_g))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->ln_1_b, &profile->ln_1_b))
    {
        return false;
    }
    t->c_attn_w = NULL;
    t->c_attn_b = NULL;
    t->q_attn_w = NULL;
    t->q_attn_b = NULL;
    t->k_attn_w = NULL;
    t->k_attn_b = NULL;
    t->v_attn_w = NULL;
    t->v_attn_b = NULL;

    if (profile->qkv_fused)
    {
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_attn_w, &profile->c_attn_w))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_attn_b, &profile->c_attn_b))
        {
            return false;
        }
    }
    else
    {
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->q_attn_w, &profile->q_attn_w))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->q_attn_b, &profile->q_attn_b))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->k_attn_w, &profile->k_attn_w))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->k_attn_b, &profile->k_attn_b))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->v_attn_w, &profile->v_attn_w))
        {
            return false;
        }
        if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->v_attn_b, &profile->v_attn_b))
        {
            return false;
        }
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_proj_w, &profile->c_proj_w))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_proj_b, &profile->c_proj_b))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->ln_2_g, &profile->ln_2_g))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->ln_2_b, &profile->ln_2_b))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_fc_w, &profile->c_fc_w))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_fc_b, &profile->c_fc_b))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_proj2_w, &profile->c_proj2_w))
    {
        return false;
    }
    if (!load_block_tensor_from_patterns(model->ctx, layer_idx, &t->c_proj2_b, &profile->c_proj2_b))
    {
        return false;
    }

    return true;
}

static struct ggml_tensor *apply_layer_norm_affine(
    struct ggml_context *ctx0,
    struct ggml_tensor *x,
    struct ggml_tensor *norm_g,
    struct ggml_tensor *norm_b)
{
    struct ggml_tensor *y = ggml_norm(ctx0, x, 1e-5f);

    if (norm_g != NULL)
    {
        y = ggml_mul(ctx0, ggml_repeat(ctx0, norm_g, y), y);
    }

    if (norm_b != NULL)
    {
        y = ggml_add(ctx0, y, ggml_repeat(ctx0, norm_b, y));
    }

    return y;
}

static struct ggml_tensor *add_bias_if_present(
    struct ggml_context *ctx0,
    struct ggml_tensor *x,
    struct ggml_tensor *bias)
{
    if (bias == NULL)
    {
        return x;
    }
    return ggml_add(ctx0, x, ggml_repeat(ctx0, bias, x));
}

static struct ggml_tensor *build_self_attention_single_token(
    struct ggml_context *ctx0,
    struct ggml_tensor *ln_1,
    const struct inference_layer_tensors *layer,
    const struct inference_profile *profile,
    int64_t n_embd,
    int64_t n_head,
    int n_ctx,
    int n_past,
    int64_t N,
    int layer_idx,
    struct ggml_tensor *memory_k,
    struct ggml_tensor *memory_v,
    struct ggml_cgraph *gf)
{
    const int64_t head_dim = n_embd / n_head;
    struct ggml_tensor *Qcur = NULL;
    struct ggml_tensor *Kcur = NULL;
    struct ggml_tensor *Vcur = NULL;

    if (profile->qkv_fused)
    {
        struct ggml_tensor *qkv = mul_mat(ctx0, layer->c_attn_w, ln_1, "c_attn_w * ln_1");
        if (qkv == NULL)
        {
            return NULL;
        }
        qkv = add_bias_if_present(ctx0, qkv, layer->c_attn_b);

        const size_t qkv_elem_size = ggml_element_size(qkv);
        Qcur = ggml_view_2d(ctx0, qkv, n_embd, N, qkv->nb[1], 0 * qkv_elem_size * n_embd);
        Kcur = ggml_view_2d(ctx0, qkv, n_embd, N, qkv->nb[1], 1 * qkv_elem_size * n_embd);
        Vcur = ggml_view_2d(ctx0, qkv, n_embd, N, qkv->nb[1], 2 * qkv_elem_size * n_embd);
    }
    else
    {
        Qcur = mul_mat(ctx0, layer->q_attn_w, ln_1, "q_attn_w * ln_1");
        Kcur = mul_mat(ctx0, layer->k_attn_w, ln_1, "k_attn_w * ln_1");
        Vcur = mul_mat(ctx0, layer->v_attn_w, ln_1, "v_attn_w * ln_1");
        if (Qcur == NULL || Kcur == NULL || Vcur == NULL)
        {
            return NULL;
        }

        Qcur = add_bias_if_present(ctx0, Qcur, layer->q_attn_b);
        Kcur = add_bias_if_present(ctx0, Kcur, layer->k_attn_b);
        Vcur = add_bias_if_present(ctx0, Vcur, layer->v_attn_b);
    }

    // Store current K/V into cache, matching the reference GPT-2 graph behavior.
    if (memory_k != NULL && memory_v != NULL && gf != NULL && N >= 1)
    {
        const size_t k_stride = ggml_element_size(memory_k) * n_embd;
        const size_t v_stride = ggml_element_size(memory_v) * n_embd;
        const int64_t layer_base = (int64_t)layer_idx * n_ctx;

        struct ggml_tensor *k_dst = ggml_view_1d(
            ctx0,
            memory_k,
            N * n_embd,
            k_stride * (layer_base + n_past));
        struct ggml_tensor *v_dst = ggml_view_1d(
            ctx0,
            memory_v,
            N * n_embd,
            v_stride * (layer_base + n_past));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_dst));
    }

    // Match reference graph: Q branch uses a single CONT_3D node before permute.
    struct ggml_tensor *Q =
        ggml_permute(ctx0,
                     ggml_cont_3d(ctx0, Qcur, head_dim, n_head, N),
                     0, 2, 1, 3);

    struct ggml_tensor *K =
        ggml_permute(ctx0,
                     ggml_reshape_3d(
                         ctx0,
                         ggml_view_1d(
                             ctx0,
                             memory_k,
                             ((int64_t)n_past + N) * n_embd,
                             (size_t)ggml_element_size(memory_k) * n_embd * ((int64_t)layer_idx * n_ctx)),
                         head_dim,
                         n_head,
                         (int64_t)n_past + N),
                     0, 2, 1, 3);

    // ggml reference GPT-2 path uses K * Q to produce [seq_k, seq_q, n_head].
    struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
    struct ggml_tensor *KQ_scaled = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float)n_embd / (float)n_head));
    // Keep causal behavior explicit even in single-token mode.
    struct ggml_tensor *KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
    struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

    struct ggml_tensor *V_trans =
        ggml_cont_3d(ctx0,
                     ggml_permute(ctx0,
                                  ggml_reshape_3d(
                                      ctx0,
                                      ggml_view_1d(
                                          ctx0,
                                          memory_v,
                                          ((int64_t)n_past + N) * n_embd,
                                          (size_t)ggml_element_size(memory_v) * n_embd * ((int64_t)layer_idx * n_ctx)),
                                      head_dim,
                                      n_head,
                                      (int64_t)n_past + N),
                                  1, 2, 0, 3),
                     (int64_t)n_past + N, head_dim, n_head);

    struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);
    struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
    struct ggml_tensor *attn_ctx = ggml_cont_2d(ctx0, KQV_merged, n_embd, N);

    struct ggml_tensor *attn_out = mul_mat(ctx0, layer->c_proj_w, attn_ctx, "c_proj_w * attn_ctx");
    if (attn_out == NULL)
    {
        return NULL;
    }
    return add_bias_if_present(ctx0, attn_out, layer->c_proj_b);
}

static struct ggml_tensor *build_ffn(
    struct ggml_context *ctx0,
    struct ggml_tensor *ln_2,
    const struct inference_layer_tensors *layer)
{
    struct ggml_tensor *ff = mul_mat(ctx0, layer->c_fc_w, ln_2, "c_fc_w * ln_2");
    if (ff == NULL)
    {
        return NULL;
    }
    ff = add_bias_if_present(ctx0, ff, layer->c_fc_b);
    ff = ggml_gelu(ctx0, ff);

    ff = mul_mat(ctx0, layer->c_proj2_w, ff, "c_proj2_w * ff");
    if (ff == NULL)
    {
        return NULL;
    }

    return add_bias_if_present(ctx0, ff, layer->c_proj2_b);
}
static struct ggml_tensor *apply_rmsnorm_affine(
    struct ggml_context *ctx0,
    struct ggml_tensor *x,
    struct ggml_tensor *weight) // LLaMA n'a pas de bias pour RMSNorm
{
    // epsilon typique LLaMA
    const float eps = 1e-6f;

    // RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    struct ggml_tensor *y = ggml_rms_norm(ctx0, x, eps);
    
    if (weight != NULL)
        y = ggml_mul(ctx0, ggml_repeat(ctx0, weight, y), y);

    return y;
}
static struct ggml_tensor *run_transformer_block_single_token(
    struct ggml_context *ctx0,
    struct ggml_tensor *x,
    const struct inference_layer_tensors *layer,
    const struct inference_profile *profile,
    int64_t n_embd,
    int64_t n_head,
    int n_ctx,
    int n_past,
    int64_t N,
    int layer_idx,
    struct ggml_tensor *memory_k,
    struct ggml_tensor *memory_v,
    struct ggml_cgraph *gf)
{
    struct ggml_tensor *ln_1 = apply_layer_norm_affine(ctx0, x, layer->ln_1_g, layer->ln_1_b);
    struct ggml_tensor *attn_out = build_self_attention_single_token(ctx0, ln_1, layer, profile, n_embd, n_head, n_ctx, n_past, N, layer_idx, memory_k, memory_v, gf);
    if (attn_out == NULL)
    {
        return NULL;
    }

    struct ggml_tensor *x_attn = ggml_add(ctx0, x, attn_out);
    struct ggml_tensor *ln_2 = apply_layer_norm_affine(ctx0, x_attn, layer->ln_2_g, layer->ln_2_b);
    struct ggml_tensor *ff = build_ffn(ctx0, ln_2, layer);
    if (ff == NULL)
    {
        return NULL;
    }

    return ggml_add(ctx0, x_attn, ff);
}

static struct ggml_tensor *build_output_logits(
    struct ggml_context *ctx0,
    struct ggml_tensor *x,
    const struct inference_global_tensors *g)
{
    struct ggml_tensor *x_norm = apply_layer_norm_affine(ctx0, x, g->ln_f_g, g->ln_f_b);
    return mul_mat(ctx0, g->lm_head, x_norm, "lm_head * x");
}

static int env_get_int_or_default(const char *name, int default_value)
{
    const char *v = getenv(name);
    if (v == NULL || *v == '\0')
    {
        return default_value;
    }

    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || *end != '\0')
    {
        return default_value;
    }

    return (int)parsed;
}

static double env_get_double_or_default(const char *name, double default_value)
{
    const char *v = getenv(name);
    if (v == NULL || *v == '\0')
    {
        return default_value;
    }

    char *end = NULL;
    double parsed = strtod(v, &end);
    if (end == v || *end != '\0')
    {
        return default_value;
    }

    return parsed;
}

static int find_graph_node_index(struct ggml_tensor **nodes, int n_nodes, const struct ggml_tensor *target)
{
    for (int i = 0; i < n_nodes; ++i)
    {
        if (nodes[i] == target)
        {
            return i;
        }
    }
    return -1;
}

static double estimate_node_compute_cost(const struct ggml_tensor *t)
{
    if (t == NULL)
    {
        return 0.0;
    }

    double cost = (double)ggml_nbytes(t);
    const double ne = (double)ggml_nelements(t);

    switch (t->op)
    {
    case GGML_OP_MUL_MAT:
        if (t->src[0] != NULL)
        {
            const double k = (double)t->src[0]->ne[0];
            cost += 0.002 * ne * (k > 1.0 ? k : 1.0);
        }
        else
        {
            cost += 4.0 * ne;
        }
        break;
    case GGML_OP_SOFT_MAX:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_RMS_NORM:
    case GGML_OP_NORM:
    case GGML_OP_UNARY:
        cost += 0.5 * ne;
        break;
    default:
        cost += 0.1 * ne;
        break;
    }

    return cost;
}

static bool node_touches_kv_cache(const struct ggml_tensor *t, const struct ggml_tensor *memory_k, const struct ggml_tensor *memory_v)
{
    if (t == NULL)
    {
        return false;
    }

    if (t == memory_k || t == memory_v || t->view_src == memory_k || t->view_src == memory_v)
    {
        return true;
    }

    for (int s = 0; s < GGML_MAX_SRC; ++s)
    {
        const struct ggml_tensor *src = t->src[s];
        if (src == NULL)
        {
            continue;
        }
        if (src == memory_k || src == memory_v || src->view_src == memory_k || src->view_src == memory_v)
        {
            return true;
        }
    }

    return false;
}

static void mark_attention_kv_hot_nodes(
    struct ggml_tensor **nodes,
    int n_nodes,
    const struct ggml_tensor *memory_k,
    const struct ggml_tensor *memory_v,
    bool *hot)
{
    for (int i = 0; i < n_nodes; ++i)
    {
        struct ggml_tensor *t = nodes[i];
        hot[i] = node_touches_kv_cache(t, memory_k, memory_v) ||
                 t->op == GGML_OP_DIAG_MASK_INF ||
                 t->op == GGML_OP_SOFT_MAX;
    }

    // One-hop expansion keeps the whole attention subgraph together around KV cache access.
    for (int i = 0; i < n_nodes; ++i)
    {
        if (hot[i])
        {
            continue;
        }

        struct ggml_tensor *t = nodes[i];
        bool near_hot = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            if (t->src[s] == NULL)
            {
                continue;
            }
            int src_idx = find_graph_node_index(nodes, n_nodes, t->src[s]);
            if (src_idx >= 0 && hot[src_idx])
            {
                near_hot = true;
                break;
            }
        }

        if (near_hot && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_SCALE || t->op == GGML_OP_PERMUTE ||
                         t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_CONT))
        {
            hot[i] = true;
        }
    }
}

static double compute_cross_partition_comm_bytes(const dist_graph_t *graph, const int *part_of)
{
    double total = 0.0;
    for (int e = 0; e < graph->n_edges; ++e)
    {
        const int src_p = part_of[graph->edges[e].src];
        const int dst_p = part_of[graph->edges[e].dst];
        if (src_p != dst_p)
        {
            total += graph->edges[e].weight;
        }
    }
    return total;
}

static double load_penalty_single(double load, double target)
{
    const double denom = target > 1e-12 ? target : 1.0;
    const double x = (load - target) / denom;
    return x * x;
}

static void apply_attention_kv_colocation_pass(
    const dist_graph_t *graph,
    const dist_partition_config_t *cfg,
    const bool *hot,
    const double *targets,
    int *part_of,
    double *loads)
{
    int anchor = 0;
    double best_hot_load = -1.0;

    for (int p = 0; p < cfg->n_partitions; ++p)
    {
        double hot_load = 0.0;
        for (int v = 0; v < graph->n_vertices; ++v)
        {
            if (hot[v] && part_of[v] == p)
            {
                hot_load += graph->vertex_compute_cost[v];
            }
        }
        if (hot_load > best_hot_load)
        {
            best_hot_load = hot_load;
            anchor = p;
        }
    }

    for (int v = 0; v < graph->n_vertices; ++v)
    {
        if (!hot[v])
        {
            continue;
        }

        const int from = part_of[v];
        if (from == anchor)
        {
            continue;
        }

        const double vc = graph->vertex_compute_cost[v];
        if (cfg->hard_cap_ratio >= 0.0)
        {
            const double max_anchor = targets[anchor] * (1.0 + cfg->hard_cap_ratio);
            if (loads[anchor] + vc > max_anchor + 1e-12)
            {
                continue;
            }
        }

        part_of[v] = anchor;
        loads[from] -= vc;
        loads[anchor] += vc;
    }
}

static void maybe_partition_runtime_graph(
    struct ggml_cgraph *gf,
    struct ggml_tensor *memory_k,
    struct ggml_tensor *memory_v)
{
    const int n_partitions = env_get_int_or_default("GGML_DISTRIB_NODES", 1);
    if (n_partitions <= 1)
    {
        return;
    }

    const int n_nodes = ggml_graph_n_nodes(gf);
    if (n_nodes <= 0)
    {
        return;
    }

    struct ggml_tensor **nodes = (struct ggml_tensor **)malloc((size_t)n_nodes * sizeof(struct ggml_tensor *));
    double *vertex_cost = (double *)malloc((size_t)n_nodes * sizeof(double));
    int *part_of = (int *)malloc((size_t)n_nodes * sizeof(int));
    double *loads = (double *)calloc((size_t)n_partitions, sizeof(double));
    bool *hot = (bool *)calloc((size_t)n_nodes, sizeof(bool));
    double *targets = (double *)malloc((size_t)n_partitions * sizeof(double));
    if (nodes == NULL || vertex_cost == NULL || part_of == NULL || loads == NULL || hot == NULL || targets == NULL)
    {
        free(nodes);
        free(vertex_cost);
        free(part_of);
        free(loads);
        free(hot);
        free(targets);
        return;
    }

    for (int i = 0; i < n_nodes; ++i)
    {
        nodes[i] = ggml_graph_node(gf, i);
        vertex_cost[i] = estimate_node_compute_cost(nodes[i]);
    }

    mark_attention_kv_hot_nodes(nodes, n_nodes, memory_k, memory_v, hot);

    const int max_edges = n_nodes * GGML_MAX_SRC;
    dist_edge_t *edges = (dist_edge_t *)malloc((size_t)max_edges * sizeof(dist_edge_t));
    if (edges == NULL)
    {
        free(nodes);
        free(vertex_cost);
        free(part_of);
        free(loads);
        free(hot);
        free(targets);
        return;
    }

    int n_edges = 0;
    for (int i = 0; i < n_nodes; ++i)
    {
        struct ggml_tensor *dst = nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            struct ggml_tensor *src = dst->src[s];
            if (src == NULL)
            {
                continue;
            }

            int src_idx = find_graph_node_index(nodes, n_nodes, src);
            if (src_idx < 0)
            {
                continue;
            }

            double w = (double)ggml_nbytes(src);
            if (w <= 0.0)
            {
                w = (double)ggml_nbytes(dst);
            }
            if (w <= 0.0)
            {
                w = 1.0;
            }

            if (hot[src_idx] && hot[i])
            {
                w *= 4.0;
            }
            else if (hot[src_idx] || hot[i])
            {
                w *= 2.0;
            }

            edges[n_edges].src = src_idx;
            edges[n_edges].dst = i;
            edges[n_edges].weight = w;
            ++n_edges;
        }
    }

    dist_graph_t graph = {
        .n_vertices = n_nodes,
        .n_edges = n_edges,
        .vertex_compute_cost = vertex_cost,
        .edges = edges,
    };

    double total_load = 0.0;
    for (int i = 0; i < n_nodes; ++i)
    {
        total_load += vertex_cost[i];
    }
    const double target = total_load / (double)n_partitions;
    for (int p = 0; p < n_partitions; ++p)
    {
        targets[p] = target;
    }

    dist_partition_config_t cfg = {
        .n_partitions = n_partitions,
        .lambda_comm = env_get_double_or_default("GGML_DISTRIB_LAMBDA_COMM", 1.0),
        .lambda_balance = env_get_double_or_default("GGML_DISTRIB_LAMBDA_BALANCE", 0.20),
        .target_partition_load = NULL,
        .hard_cap_ratio = env_get_double_or_default("GGML_DISTRIB_HARD_CAP_RATIO", 0.25),
        .max_refine_iters = env_get_int_or_default("GGML_DISTRIB_MAX_ITERS", 20),
        .min_delta_improvement = env_get_double_or_default("GGML_DISTRIB_MIN_DELTA", 1e-9),
    };

    dist_partition_result_t result = {
        .vertex_to_partition = part_of,
        .partition_loads = loads,
        .cut_cost = 0.0,
        .balance_penalty = 0.0,
        .objective = 0.0,
    };

    bool partition_ok = false;
    const int use_topology_tree = env_get_int_or_default("GGML_DISTRIB_TOPOLOGY_TREE", 0);
    if (use_topology_tree != 0)
    {
        dist_topology_tree_t tree = {0};
        if (dist_build_topology_tree_from_cgraph(gf, &tree))
        {
            if (dist_partition_topology_tree_levels(&tree, n_partitions, result.vertex_to_partition, result.partition_loads))
            {
                result.cut_cost = dist_topology_tree_cut_cost(&tree, result.vertex_to_partition);

                double tree_balance_penalty = 0.0;
                for (int p = 0; p < n_partitions; ++p)
                {
                    tree_balance_penalty += load_penalty_single(result.partition_loads[p], targets[p]);
                }

                result.balance_penalty = tree_balance_penalty;
                result.objective = cfg.lambda_comm * result.cut_cost + cfg.lambda_balance * result.balance_penalty;
                partition_ok = true;

                if (env_get_int_or_default("GGML_DISTRIB_VERBOSE", 0) != 0)
                {
                    dist_print_topology_tree(&tree, stdout);
                }
            }

            dist_free_topology_tree(&tree);
        }

        if (!partition_ok)
        {
            printf("[distrib] topology tree partitioning failed, falling back to min-cut\n");
        }
    }

    if (!partition_ok)
    {
        if (!dist_partition_graph_balanced_min_cut(&graph, &cfg, &result))
        {
            printf("[distrib] Partitioning failed\n");
            free(nodes);
            free(vertex_cost);
            free(part_of);
            free(loads);
            free(hot);
            free(targets);
            free(edges);
            return;
        }

        const int colocate_kv = env_get_int_or_default("GGML_DISTRIB_COLOCATE_KV", 1);
        if (colocate_kv != 0)
        {
            apply_attention_kv_colocation_pass(&graph, &cfg, hot, targets, result.vertex_to_partition, result.partition_loads);
        }
    }

    const double cut_after_colocation = compute_cross_partition_comm_bytes(&graph, result.vertex_to_partition);
    const int verbose = env_get_int_or_default("GGML_DISTRIB_VERBOSE", 0);

    printf("[distrib] n_nodes=%d n_edges=%d partitions=%d comm_cut_bytes=%.0f objective=%.6f\n",
           graph.n_vertices,
           graph.n_edges,
           cfg.n_partitions,
           cut_after_colocation,
           result.objective);

    if (verbose != 0)
    {
        for (int p = 0; p < cfg.n_partitions; ++p)
        {
            printf("[distrib] partition[%d] load=%.2f (target=%.2f)\n", p, result.partition_loads[p], targets[p]);
        }

        const int max_print = n_nodes < 64 ? n_nodes : 64;
        for (int i = 0; i < max_print; ++i)
        {
            const struct ggml_tensor *t = nodes[i];
            printf("[distrib] node[%d] op=%s part=%d hot=%d ne=[%lld,%lld,%lld,%lld]\n",
                   i,
                   ggml_op_name(t->op),
                   result.vertex_to_partition[i],
                   hot[i] ? 1 : 0,
                   (long long)t->ne[0],
                   (long long)t->ne[1],
                   (long long)t->ne[2],
                   (long long)t->ne[3]);
        }
    }

    // Export to DOT if requested
    const char *dot_export_env = getenv("GGML_DISTRIB_DOT_EXPORT");
    if (dot_export_env != NULL && atoi(dot_export_env) != 0)
    {
        const char **vertex_labels = (const char **)malloc((size_t)n_nodes * sizeof(const char *));
        if (vertex_labels != NULL)
        {
            for (int i = 0; i < n_nodes; ++i)
            {
                vertex_labels[i] = ggml_op_name(nodes[i]->op);
            }

            // Pass NULL to use built-in default names for all partitions.
            // A single-entry array would be out-of-bounds when n_partitions > 1.
            dist_export_partition_to_dot("partition-plan.dot", &graph, &cfg, &result,
                                        vertex_labels, NULL);
            printf("[distrib] DOT graph exported to partition-plan.dot\n");

            free(vertex_labels);
        }
    }

    free(nodes);
    free(vertex_cost);
    free(part_of);
    free(loads);
    free(hot);
    free(targets);
    free(edges);
}

bool run_batch_inference(struct model *model, const int *token_ids, int n_tokens, int n_past)
{
    if (token_ids == NULL || n_tokens <= 0)
    {
        printf("Invalid batch input: token_ids=%p n_tokens=%d\n", (void *)token_ids, n_tokens);
        return false;
    }

    const struct inference_profile *profile = get_inference_profile(model);
    printf("Using inference profile: %s (architecture=%s)\n",
           profile->profile_name,
           model->model_name != NULL ? model->model_name : "unknown");

    struct inference_global_tensors g = {0};
    if (!load_inference_global_tensors(model, profile, &g))
    {
        return false;
    }

    int64_t vocab_size = model->hparams.n_vocab;
    if (vocab_size <= 0)
    {
        vocab_size = g.wte->ne[1];
    }

    for (int i = 0; i < n_tokens; ++i)
    {
        if (token_ids[i] < 0 || (int64_t)token_ids[i] >= vocab_size)
        {
            printf("Token id out of range at batch index %d: %d (vocab size: %lld)\n",
                   i, token_ids[i], (long long)vocab_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    if (ctx0 == NULL)
    {
        printf("Failed to init inference context\n");
        return false;
    }

    const int n_layer = model->hparams.n_layer;
    const int n_ctx = model->hparams.n_ctx;
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_head = model->hparams.n_head;

    if (n_past < 0 || n_past + n_tokens > n_ctx)
    {
        printf("Invalid sequence window: n_tokens=%d n_past=%d n_ctx=%d\n", n_tokens, n_past, n_ctx);
        ggml_free(ctx0);
        return false;
    }

    if (n_head <= 0 || (n_embd % n_head) != 0)
    {
        printf("Invalid attention dimensions: n_embd=%lld n_head=%lld\n",
               (long long)n_embd, (long long)n_head);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor *token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; ++i)
    {
        ((int32_t *)token->data)[i] = token_ids[i];
    }

    struct ggml_tensor *position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; ++i)
    {
        ((int32_t *)position->data)[i] = n_past + i;
    }

    struct ggml_tensor *x_tok = ggml_get_rows(ctx0, g.wte, token);
    struct ggml_tensor *x = x_tok;

    if (g.wpe != NULL)
    {
        x = ggml_add(ctx0, x_tok, ggml_get_rows(ctx0, g.wpe, position));
    }
    else
    {
        printf("No absolute position embedding tensor found; using token embeddings only.\n");
    }

    struct ggml_tensor *memory_k = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)n_layer * n_ctx * n_embd);
    struct ggml_tensor *memory_v = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)n_layer * n_ctx * n_embd);
    memset(memory_k->data, 0, ggml_nbytes(memory_k));
    memset(memory_v->data, 0, ggml_nbytes(memory_v));

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    for (int i = 0; i < n_layer; ++i)
    {
        struct inference_layer_tensors layer = {0};
        if (!load_inference_layer_tensors(model, profile, i, &layer))
        {
            ggml_free(ctx0);
            return false;
        }

        x = run_transformer_block_single_token(ctx0, x, &layer, profile, n_embd, n_head, n_ctx, n_past, n_tokens, i, memory_k, memory_v, gf);
        if (x == NULL)
        {
            ggml_free(ctx0);
            return false;
        }
    }

    struct ggml_tensor *logits = build_output_logits(ctx0, x, &g);
    if (logits == NULL)
    {
        ggml_free(ctx0);
        return false;
    }

    if (logits->type != GGML_TYPE_F32)
    {
        logits = ggml_cast(ctx0, logits, GGML_TYPE_F32);
    }

    ggml_build_forward_expand(gf, logits);
    maybe_partition_runtime_graph(gf, memory_k, memory_v);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_free(backend);

    int n_logits = (int)ggml_nelements(logits);
    int max_logits = model->hparams.n_vocab * n_tokens;
    if (n_logits > max_logits)
    {
        n_logits = max_logits;
    }

    save_logits((float *)logits->data, n_logits);
    printf("Batch inference done (n_tokens=%d n_past=%d)\n", n_tokens, n_past);

    //ggml_graph_print(gf);
    //ggml_graph_dump_dot(gf, NULL, "debug.dot");

    ggml_free(ctx0);
    return true;
}

bool run_single_token_inference(struct model *model, int token_id)
{
    int n_tokens = 1;
    int n_past = 0;

    int *token_ids = (int *)malloc((size_t)n_tokens * sizeof(int));
    if (token_ids == NULL)
    {
        printf("Failed to allocate token batch\n");
        return false;
    }

    for (int i = 0; i < n_tokens; ++i)
    {
        token_ids[i] = token_id;
    }

    bool ok = run_batch_inference(model, token_ids, n_tokens, n_past);
    free(token_ids);
    return ok;
}
