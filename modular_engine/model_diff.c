#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "parser.h"

static struct ggml_tensor *find_tensor_any(
    struct ggml_context *ctx,
    const char *const *names,
    int n_names,
    const char **matched_name)
{
    for (int i = 0; i < n_names; ++i) {
        struct ggml_tensor *t = ggml_get_tensor(ctx, names[i]);
        if (t != NULL) {
            if (matched_name != NULL) {
                *matched_name = names[i];
            }
            return t;
        }
    }

    if (matched_name != NULL) {
        *matched_name = NULL;
    }
    return NULL;
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
    if (*out == NULL) {
        printf("Tensor not found for %s. Tried:\n", label);
        for (int i = 0; i < n_names; ++i) {
            printf("  - %s\n", names[i]);
        }
        return false;
    }
    return true;
}

struct inference_layer_tensors {
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;
    struct ggml_tensor *c_attn_w;
    struct ggml_tensor *c_attn_b;
    struct ggml_tensor *c_proj_w;
    struct ggml_tensor *c_proj_b;
    struct ggml_tensor *ln_2_g;
    struct ggml_tensor *ln_2_b;
    struct ggml_tensor *c_fc_w;
    struct ggml_tensor *c_fc_b;
    struct ggml_tensor *c_proj2_w;
    struct ggml_tensor *c_proj2_b;
};

struct inference_global_tensors {
    struct ggml_tensor *wte;
    struct ggml_tensor *wpe;
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;
    struct ggml_tensor *lm_head;
};

struct diff_stats {
    int layer_diffs;
    double max_abs_diff;
    int max_diff_layer;
    const char *max_diff_stage;
};

static bool compare_f32_tensors(
    const struct ggml_tensor *t_ref,
    const struct ggml_tensor *t_custom,
    const char *label,
    struct diff_stats *stats,
    double threshold)
{
    if (t_ref == NULL || t_custom == NULL) {
        if (t_ref != t_custom) {
            printf("[%s] MISMATCH: one tensor is NULL\n", label);
            return false;
        }
        return true;
    }

    if (t_ref->type != GGML_TYPE_F32 || t_custom->type != GGML_TYPE_F32) {
        return true;
    }

    const int64_t n = ggml_nelements(t_ref);
    if (n != ggml_nelements(t_custom)) {
        printf("[%s] Element count mismatch: %lld vs %lld\n", label, (long long)n, (long long)ggml_nelements(t_custom));
        return false;
    }

    const float *pref = (const float *)t_ref->data;
    const float *pcus = (const float *)t_custom->data;

    double max_diff = 0.0;
    double mean_diff = 0.0;
    int count_above_threshold = 0;

    for (int64_t i = 0; i < n; ++i) {
        double d = fabs((double)pref[i] - (double)pcus[i]);
        mean_diff += d;
        if (d > max_diff) {
            max_diff = d;
        }
        if (d > threshold) {
            count_above_threshold++;
        }
    }
    mean_diff /= (double)n;

    if (max_diff > threshold) {
        printf("[%s] Value diff detected:\n", label);
        printf("  max_abs_diff: %.8g, mean_diff: %.8g\n", max_diff, mean_diff);
        printf("  Elements above threshold (%.2e): %d / %lld (%.1f%%)\n",
               threshold, count_above_threshold, (long long)n, 100.0 * count_above_threshold / n);

        if (max_diff > stats->max_abs_diff) {
            stats->max_abs_diff = max_diff;
            stats->max_diff_stage = label;
        }
        return false;
    }

    return true;
}

bool run_inference_with_capture(struct model *model, int token_id, struct ggml_tensor **out_logits)
{
    const struct model *profile_model = model;

    struct inference_global_tensors g = {0};
    {
        const char *names_wte[] = {"token_embd.weight", "transformer.wte.weight"};
        if (!load_required_tensor_any(model->ctx, &g.wte, names_wte, 2, "wte")) {
            return false;
        }
        const char *names_wpe[] = {"position_embd.weight", "transformer.wpe.weight"};
        if (!load_required_tensor_any(model->ctx, &g.wpe, names_wpe, 2, "wpe")) {
            return false;
        }
        const char *names_ln_f_g[] = {"output_norm.weight", "transformer.ln_f.weight"};
        if (!load_required_tensor_any(model->ctx, &g.ln_f_g, names_ln_f_g, 2, "ln_f.weight")) {
            return false;
        }
        const char *names_ln_f_b[] = {"output_norm.bias", "transformer.ln_f.bias"};
        if (!load_required_tensor_any(model->ctx, &g.ln_f_b, names_ln_f_b, 2, "ln_f.bias")) {
            return false;
        }
        const char *names_lm_head[] = {"output.weight", "lm_head.weight"};
        if (!load_required_tensor_any(model->ctx, &g.lm_head, names_lm_head, 2, "lm_head")) {
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    if (ctx0 == NULL) {
        printf("Failed to init inference context\n");
        return false;
    }

    struct ggml_tensor *token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ((int32_t *)token->data)[0] = token_id;

    struct ggml_tensor *position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ((int32_t *)position->data)[0] = 0;

    struct ggml_tensor *x = ggml_add(
        ctx0,
        ggml_get_rows(ctx0, g.wte, token),
        ggml_get_rows(ctx0, g.wpe, position));

    const int n_layer = model->hparams.n_layer;
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_head = model->hparams.n_head;

    if (n_head <= 0 || (n_embd % n_head) != 0) {
        printf("Invalid attention dimensions: n_embd=%lld n_head=%lld\n",
               (long long)n_embd, (long long)n_head);
        ggml_free(ctx0);
        return false;
    }

    for (int layer_idx = 0; layer_idx < n_layer; ++layer_idx) {
        struct inference_layer_tensors layer = {0};

        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_norm.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.ln_1.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.ln_1_g, names, 2, "layer.ln_1_g")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_norm.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.ln_1.bias", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.ln_1_b, names, 2, "layer.ln_1_b")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_qkv.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.attn.c_attn.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.c_attn_w, names, 2, "layer.c_attn_w")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128], n3[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_qkv.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.attn.c_attn.bias", layer_idx);
            snprintf(n3, sizeof(n3), "transformer.h.%d.attn.c_attn.b", layer_idx);
            const char *names[] = {n1, n2, n3};
            if (!load_required_tensor_any(model->ctx, &layer.c_attn_b, names, 3, "layer.c_attn_b")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_output.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.attn.c_proj.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.c_proj_w, names, 2, "layer.c_proj_w")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128], n3[128];
            snprintf(n1, sizeof(n1), "blk.%d.attn_output.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.attn.c_proj.bias", layer_idx);
            snprintf(n3, sizeof(n3), "transformer.h.%d.attn.c_proj.b", layer_idx);
            const char *names[] = {n1, n2, n3};
            if (!load_required_tensor_any(model->ctx, &layer.c_proj_b, names, 3, "layer.c_proj_b")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_norm.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.ln_2.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.ln_2_g, names, 2, "layer.ln_2_g")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128], n3[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_norm.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.ln_2.bias", layer_idx);
            snprintf(n3, sizeof(n3), "transformer.h.%d.ln_2.b", layer_idx);
            const char *names[] = {n1, n2, n3};
            if (!load_required_tensor_any(model->ctx, &layer.ln_2_b, names, 3, "layer.ln_2_b")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_up.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.mlp.c_fc.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.c_fc_w, names, 2, "layer.c_fc_w")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128], n3[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_up.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.mlp.c_fc.bias", layer_idx);
            snprintf(n3, sizeof(n3), "transformer.h.%d.mlp.c_fc.b", layer_idx);
            const char *names[] = {n1, n2, n3};
            if (!load_required_tensor_any(model->ctx, &layer.c_fc_b, names, 3, "layer.c_fc_b")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_down.weight", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.mlp.c_proj.weight", layer_idx);
            const char *names[] = {n1, n2};
            if (!load_required_tensor_any(model->ctx, &layer.c_proj2_w, names, 2, "layer.c_proj2_w")) {
                ggml_free(ctx0);
                return false;
            }
        }
        {
            char n1[128], n2[128], n3[128];
            snprintf(n1, sizeof(n1), "blk.%d.ffn_down.bias", layer_idx);
            snprintf(n2, sizeof(n2), "transformer.h.%d.mlp.c_proj.bias", layer_idx);
            snprintf(n3, sizeof(n3), "transformer.h.%d.mlp.c_proj.b", layer_idx);
            const char *names[] = {n1, n2, n3};
            if (!load_required_tensor_any(model->ctx, &layer.c_proj2_b, names, 3, "layer.c_proj2_b")) {
                ggml_free(ctx0);
                return false;
            }
        }

        struct ggml_tensor *ln_1 = ggml_norm(ctx0, x, 1e-5f);
        if (layer.ln_1_g != NULL) {
            ln_1 = ggml_mul(ctx0, ggml_repeat(ctx0, layer.ln_1_g, ln_1), ln_1);
        }
        if (layer.ln_1_b != NULL) {
            ln_1 = ggml_add(ctx0, ln_1, ggml_repeat(ctx0, layer.ln_1_b, ln_1));
        }

        struct ggml_tensor *qkv = ggml_mul_mat(ctx0, layer.c_attn_w, ln_1);
        if (layer.c_attn_b != NULL) {
            qkv = ggml_add(ctx0, qkv, ggml_repeat(ctx0, layer.c_attn_b, qkv));
        }

        struct ggml_tensor *attn_out = ggml_mul_mat(ctx0, layer.c_proj_w, qkv);
        if (layer.c_proj_b != NULL) {
            attn_out = ggml_add(ctx0, attn_out, ggml_repeat(ctx0, layer.c_proj_b, attn_out));
        }

        x = ggml_add(ctx0, x, attn_out);

        struct ggml_tensor *ln_2 = ggml_norm(ctx0, x, 1e-5f);
        if (layer.ln_2_g != NULL) {
            ln_2 = ggml_mul(ctx0, ggml_repeat(ctx0, layer.ln_2_g, ln_2), ln_2);
        }
        if (layer.ln_2_b != NULL) {
            ln_2 = ggml_add(ctx0, ln_2, ggml_repeat(ctx0, layer.ln_2_b, ln_2));
        }

        struct ggml_tensor *ff = ggml_mul_mat(ctx0, layer.c_fc_w, ln_2);
        if (layer.c_fc_b != NULL) {
            ff = ggml_add(ctx0, ff, ggml_repeat(ctx0, layer.c_fc_b, ff));
        }
        ff = ggml_gelu(ctx0, ff);

        ff = ggml_mul_mat(ctx0, layer.c_proj2_w, ff);
        if (layer.c_proj2_b != NULL) {
            ff = ggml_add(ctx0, ff, ggml_repeat(ctx0, layer.c_proj2_b, ff));
        }

        x = ggml_add(ctx0, x, ff);
    }

    struct ggml_tensor *x_norm = ggml_norm(ctx0, x, 1e-5f);
    if (g.ln_f_g != NULL) {
        x_norm = ggml_mul(ctx0, ggml_repeat(ctx0, g.ln_f_g, x_norm), x_norm);
    }
    if (g.ln_f_b != NULL) {
        x_norm = ggml_add(ctx0, x_norm, ggml_repeat(ctx0, g.ln_f_b, x_norm));
    }

    struct ggml_tensor *logits = ggml_mul_mat(ctx0, g.lm_head, x_norm);

    if (logits->type != GGML_TYPE_F32) {
        logits = ggml_cast(ctx0, logits, GGML_TYPE_F32);
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_free(backend);

    *out_logits = logits;
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s model_ref.gguf model_custom.gguf token_id [threshold]\n", argv[0]);
        printf("  Compares intermediate tensors from two models during inference.\n");
        printf("  threshold: diff tolerance (default: 1e-4)\n");
        return 1;
    }

    const char *ref_path = argv[1];
    const char *custom_path = argv[2];
    int token_id = atoi(argv[3]);
    double threshold = 1e-4;

    if (argc >= 5) {
        threshold = atof(argv[4]);
    }

    struct model ref_model = {0};
    struct model custom_model = {0};

    printf("Loading reference model from %s...\n", ref_path);
    if (!model_load(ref_path, &ref_model)) {
        printf("Failed to load reference model\n");
        return 1;
    }

    printf("Loading custom model from %s...\n", custom_path);
    if (!model_load(custom_path, &custom_model)) {
        printf("Failed to load custom model\n");
        gguf_free(ref_model.ctxgguf);
        free(ref_model.model_name);
        return 1;
    }

    printf("\n=== Models Loaded ===\n");
    printf("Reference: %s\n", ref_model.model_name);
    printf("  n_layer=%d, n_embd=%d, n_head=%d, n_vocab=%d\n",
           ref_model.hparams.n_layer, ref_model.hparams.n_embd,
           ref_model.hparams.n_head, ref_model.hparams.n_vocab);
    printf("Custom:    %s\n", custom_model.model_name);
    printf("  n_layer=%d, n_embd=%d, n_head=%d, n_vocab=%d\n",
           custom_model.hparams.n_layer, custom_model.hparams.n_embd,
           custom_model.hparams.n_head, custom_model.hparams.n_vocab);
    printf("\nToken ID: %d, Threshold: %.2e\n", token_id, threshold);

    struct ggml_tensor *ref_logits = NULL;
    struct ggml_tensor *custom_logits = NULL;

    printf("\n=== Running Reference Inference ===\n");
    if (!run_inference_with_capture(&ref_model, token_id, &ref_logits)) {
        printf("Reference inference failed\n");
        gguf_free(ref_model.ctxgguf);
        gguf_free(custom_model.ctxgguf);
        free(ref_model.model_name);
        free(custom_model.model_name);
        return 1;
    }

    printf("=== Running Custom Inference ===\n");
    if (!run_inference_with_capture(&custom_model, token_id, &custom_logits)) {
        printf("Custom inference failed\n");
        gguf_free(ref_model.ctxgguf);
        gguf_free(custom_model.ctxgguf);
        free(ref_model.model_name);
        free(custom_model.model_name);
        return 1;
    }

    printf("\n=== Comparing Logits ===\n");
    struct diff_stats stats = {0};
    stats.max_abs_diff = 0.0;

    bool logits_match = compare_f32_tensors(ref_logits, custom_logits, "LOGITS", &stats, threshold);

    printf("\n=== Summary ===\n");
    printf("Logits match: %s\n", logits_match ? "YES" : "NO");
    if (stats.max_abs_diff > 0.0) {
        printf("Largest diff stage: %s (%.8g)\n", stats.max_diff_stage, stats.max_abs_diff);
    }

    gguf_free(ref_model.ctxgguf);
    gguf_free(custom_model.ctxgguf);
    free(ref_model.model_name);
    free(custom_model.model_name);

    return logits_match ? 0 : 1;
}
