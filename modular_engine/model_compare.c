#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "gguf.h"
#include "parser.h"

struct tensor_entry {
    const char * original_name;
    char canonical_name[256];
    struct ggml_tensor * tensor;
    bool matched;
};

struct compare_stats {
    int missing_in_b;
    int extra_in_b;
    int type_mismatch;
    int shape_mismatch;
    int byte_diff;
    int f32_diff;
};

static uint64_t fnv1a_64(const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool shape_equal(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return a->ne[0] == b->ne[0] &&
           a->ne[1] == b->ne[1] &&
           a->ne[2] == b->ne[2] &&
           a->ne[3] == b->ne[3];
}

static void build_canonical_from_layer(char * out, size_t out_size, int layer, const char * suffix) {
    snprintf(out, out_size, "blk.%d.%s", layer, suffix);
}

static void canonicalize_name(const char * in, char * out, size_t out_size) {
    if (strcmp(in, "transformer.wte.weight") == 0) {
        snprintf(out, out_size, "token_embd.weight");
        return;
    }
    if (strcmp(in, "transformer.wpe.weight") == 0) {
        snprintf(out, out_size, "position_embd.weight");
        return;
    }
    if (strcmp(in, "transformer.ln_f.weight") == 0) {
        snprintf(out, out_size, "output_norm.weight");
        return;
    }
    if (strcmp(in, "transformer.ln_f.bias") == 0) {
        snprintf(out, out_size, "output_norm.bias");
        return;
    }
    if (strcmp(in, "lm_head.weight") == 0) {
        snprintf(out, out_size, "output.weight");
        return;
    }

    int layer = -1;

    if (sscanf(in, "transformer.h.%d.ln_1.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_norm.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.ln_1.bias", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_norm.bias");
        return;
    }
    if (sscanf(in, "transformer.h.%d.attn.c_attn.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_qkv.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.attn.c_attn.bias", &layer) == 1 ||
        sscanf(in, "transformer.h.%d.attn.c_attn.b", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_qkv.bias");
        return;
    }
    if (sscanf(in, "transformer.h.%d.attn.c_proj.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_output.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.attn.c_proj.bias", &layer) == 1 ||
        sscanf(in, "transformer.h.%d.attn.c_proj.b", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "attn_output.bias");
        return;
    }
    if (sscanf(in, "transformer.h.%d.ln_2.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_norm.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.ln_2.bias", &layer) == 1 ||
        sscanf(in, "transformer.h.%d.ln_2.b", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_norm.bias");
        return;
    }
    if (sscanf(in, "transformer.h.%d.mlp.c_fc.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_up.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.mlp.c_fc.bias", &layer) == 1 ||
        sscanf(in, "transformer.h.%d.mlp.c_fc.b", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_up.bias");
        return;
    }
    if (sscanf(in, "transformer.h.%d.mlp.c_proj.weight", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_down.weight");
        return;
    }
    if (sscanf(in, "transformer.h.%d.mlp.c_proj.bias", &layer) == 1 ||
        sscanf(in, "transformer.h.%d.mlp.c_proj.b", &layer) == 1) {
        build_canonical_from_layer(out, out_size, layer, "ffn_down.bias");
        return;
    }

    snprintf(out, out_size, "%s", in);
}

static int collect_entries(const struct model * model, struct tensor_entry ** out_entries) {
    const int n = model->n_tensors;
    struct tensor_entry * entries = (struct tensor_entry *) calloc((size_t) n, sizeof(struct tensor_entry));
    if (entries == NULL) {
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(model->ctxgguf, i);
        entries[i].original_name = name;
        entries[i].tensor = ggml_get_tensor(model->ctx, name);
        entries[i].matched = false;
        canonicalize_name(name, entries[i].canonical_name, sizeof(entries[i].canonical_name));
    }

    *out_entries = entries;
    return n;
}

static int find_unmatched_by_canonical(struct tensor_entry * entries, int n, const char * canonical_name) {
    for (int i = 0; i < n; ++i) {
        if (!entries[i].matched && strcmp(entries[i].canonical_name, canonical_name) == 0) {
            return i;
        }
    }
    return -1;
}

static void compare_hparams(const struct model * a, const struct model * b) {
    printf("=== Hyperparameters ===\n");
    printf("n_layer: %d vs %d%s\n", a->hparams.n_layer, b->hparams.n_layer,
           a->hparams.n_layer == b->hparams.n_layer ? "" : "  [DIFF]");
    printf("n_ctx:   %d vs %d%s\n", a->hparams.n_ctx, b->hparams.n_ctx,
           a->hparams.n_ctx == b->hparams.n_ctx ? "" : "  [DIFF]");
    printf("n_embd:  %d vs %d%s\n", a->hparams.n_embd, b->hparams.n_embd,
           a->hparams.n_embd == b->hparams.n_embd ? "" : "  [DIFF]");
    printf("n_ff:    %d vs %d%s\n", a->hparams.n_ff, b->hparams.n_ff,
           a->hparams.n_ff == b->hparams.n_ff ? "" : "  [DIFF]");
    printf("n_head:  %d vs %d%s\n", a->hparams.n_head, b->hparams.n_head,
           a->hparams.n_head == b->hparams.n_head ? "" : "  [DIFF]");
    printf("n_vocab: %d vs %d%s\n", a->hparams.n_vocab, b->hparams.n_vocab,
           a->hparams.n_vocab == b->hparams.n_vocab ? "" : "  [DIFF]");
    printf("\n");
}

static void maybe_compare_f32(const struct tensor_entry * ea,
                              const struct tensor_entry * eb,
                              struct compare_stats * stats,
                              double * global_max_abs,
                              char * global_max_name,
                              size_t global_max_name_size) {
    if (ea->tensor->type != GGML_TYPE_F32 || eb->tensor->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t n = ggml_nelements(ea->tensor);
    if (n != ggml_nelements(eb->tensor)) {
        return;
    }

    const float * pa = (const float *) ea->tensor->data;
    const float * pb = (const float *) eb->tensor->data;

    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = fabs((double) pa[i] - (double) pb[i]);
        mean_abs += d;
        if (d > max_abs) {
            max_abs = d;
        }
    }
    if (n > 0) {
        mean_abs /= (double) n;
    }

    if (max_abs > 0.0) {
        stats->f32_diff += 1;
        printf("F32 value diff: %s (max_abs=%.8g, mean_abs=%.8g)\n", ea->canonical_name, max_abs, mean_abs);
    }

    if (max_abs > *global_max_abs) {
        *global_max_abs = max_abs;
        snprintf(global_max_name, global_max_name_size, "%s", ea->canonical_name);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: %s model_a.gguf model_b.gguf [--strict]\n", argv[0]);
        return 1;
    }

    const char * model_a_path = argv[1];
    const char * model_b_path = argv[2];

    bool strict = false;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--strict") == 0) {
            strict = true;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    struct model a = {0};
    struct model b = {0};

    if (!model_load(model_a_path, &a)) {
        printf("Failed to load model A: %s\n", model_a_path);
        return 1;
    }
    if (!model_load(model_b_path, &b)) {
        printf("Failed to load model B: %s\n", model_b_path);
        gguf_free(a.ctxgguf);
        free(a.model_name);
        return 1;
    }

    printf("Model A: %s (%s)\n", model_a_path, a.model_name != NULL ? a.model_name : "unknown");
    printf("Model B: %s (%s)\n\n", model_b_path, b.model_name != NULL ? b.model_name : "unknown");

    compare_hparams(&a, &b);

    struct tensor_entry * ea = NULL;
    struct tensor_entry * eb = NULL;
    const int na = collect_entries(&a, &ea);
    const int nb = collect_entries(&b, &eb);

    if (na < 0 || nb < 0) {
        printf("Failed to allocate tensor entry tables\n");
        free(ea);
        free(eb);
        gguf_free(a.ctxgguf);
        gguf_free(b.ctxgguf);
        free(a.model_name);
        free(b.model_name);
        return 1;
    }

    struct compare_stats stats = {0};
    double global_max_abs = 0.0;
    char global_max_name[256] = {0};

    printf("=== Tensor Comparison (canonical GPT-2 names) ===\n");

    for (int i = 0; i < na; ++i) {
        int j = find_unmatched_by_canonical(eb, nb, ea[i].canonical_name);
        if (j < 0) {
            stats.missing_in_b += 1;
            printf("Missing in B: %s (from A name: %s)\n", ea[i].canonical_name, ea[i].original_name);
            continue;
        }

        ea[i].matched = true;
        eb[j].matched = true;

        const struct ggml_tensor * ta = ea[i].tensor;
        const struct ggml_tensor * tb = eb[j].tensor;

        if (ta == NULL || tb == NULL) {
            printf("Tensor pointer missing after lookup: %s\n", ea[i].canonical_name);
            stats.missing_in_b += 1;
            continue;
        }

        if (ta->type != tb->type) {
            stats.type_mismatch += 1;
            printf("Type mismatch: %s (%s vs %s)\n",
                   ea[i].canonical_name,
                   ggml_type_name(ta->type),
                   ggml_type_name(tb->type));
            continue;
        }

        if (!shape_equal(ta, tb)) {
            stats.shape_mismatch += 1;
            printf("Shape mismatch: %s ([%lld,%lld,%lld,%lld] vs [%lld,%lld,%lld,%lld])\n",
                   ea[i].canonical_name,
                   (long long) ta->ne[0], (long long) ta->ne[1], (long long) ta->ne[2], (long long) ta->ne[3],
                   (long long) tb->ne[0], (long long) tb->ne[1], (long long) tb->ne[2], (long long) tb->ne[3]);
            continue;
        }

        const size_t sa = ggml_nbytes(ta);
        const size_t sb = ggml_nbytes(tb);
        if (sa != sb) {
            stats.byte_diff += 1;
            printf("Byte-size mismatch: %s (%zu vs %zu)\n", ea[i].canonical_name, sa, sb);
            continue;
        }

        const uint64_t ha = fnv1a_64(ta->data, sa);
        const uint64_t hb = fnv1a_64(tb->data, sb);
        if (ha != hb) {
            stats.byte_diff += 1;
            printf("Content hash mismatch: %s (0x%016llx vs 0x%016llx)\n",
                   ea[i].canonical_name,
                   (unsigned long long) ha,
                   (unsigned long long) hb);
        }

        maybe_compare_f32(&ea[i], &eb[j], &stats, &global_max_abs, global_max_name, sizeof(global_max_name));
    }

    for (int j = 0; j < nb; ++j) {
        if (!eb[j].matched) {
            stats.extra_in_b += 1;
            printf("Extra in B: %s (from B name: %s)\n", eb[j].canonical_name, eb[j].original_name);
        }
    }

    printf("\n=== Summary ===\n");
    printf("A tensor count: %d\n", na);
    printf("B tensor count: %d\n", nb);
    printf("Missing in B: %d\n", stats.missing_in_b);
    printf("Extra in B: %d\n", stats.extra_in_b);
    printf("Type mismatches: %d\n", stats.type_mismatch);
    printf("Shape mismatches: %d\n", stats.shape_mismatch);
    printf("Byte/hash mismatches: %d\n", stats.byte_diff);
    printf("F32 tensors with value diffs: %d\n", stats.f32_diff);
    if (global_max_abs > 0.0) {
        printf("Largest F32 abs diff: %.8g (%s)\n", global_max_abs, global_max_name);
    }

    const int total_issues = stats.missing_in_b + stats.extra_in_b +
                             stats.type_mismatch + stats.shape_mismatch + stats.byte_diff;

    free(ea);
    free(eb);
    gguf_free(a.ctxgguf);
    gguf_free(b.ctxgguf);
    free(a.model_name);
    free(b.model_name);

    if (strict && total_issues > 0) {
        return 2;
    }

    return 0;
}
