#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "gguf.h"
#include "parser.h"

static int parse_non_negative_int(const char *s, int *out) {
    if (s == NULL || *s == '\0') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0) {
        return 0;
    }

    *out = (int)v;
    return 1;
}

static int parse_token_csv(const char *csv, int **out_tokens, int *out_count) {
    if (csv == NULL || *csv == '\0') {
        return 0;
    }

    char *buf = strdup(csv);
    if (buf == NULL) {
        return 0;
    }

    int capacity = 16;
    int count = 0;
    int *tokens = (int *)malloc((size_t)capacity * sizeof(int));
    if (tokens == NULL) {
        free(buf);
        return 0;
    }

    char *saveptr = NULL;
    for (char *part = strtok_r(buf, ",", &saveptr); part != NULL; part = strtok_r(NULL, ",", &saveptr)) {
        int tok = 0;
        if (!parse_non_negative_int(part, &tok)) {
            free(tokens);
            free(buf);
            return 0;
        }

        if (count == capacity) {
            capacity *= 2;
            int *grown = (int *)realloc(tokens, (size_t)capacity * sizeof(int));
            if (grown == NULL) {
                free(tokens);
                free(buf);
                return 0;
            }
            tokens = grown;
        }

        tokens[count++] = tok;
    }

    free(buf);

    if (count <= 0) {
        free(tokens);
        return 0;
    }

    *out_tokens = tokens;
    *out_count = count;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s model.gguf token_id_or_csv [n_past]\n", argv[0]);
        printf("  Single token: %s model.gguf 1\n", argv[0]);
        printf("  Batch tokens: %s model.gguf 1,2,3,4,5 0\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    const char *token_arg = argv[2];
    int n_past = 0;
    const bool n_past_from_cli = (argc >= 4);

    if (argc >= 4 && !parse_non_negative_int(argv[3], &n_past)) {
        printf("Invalid n_past: %s\n", argv[3]);
        return 1;
    }

    int *tokens = NULL;
    int n_tokens = 0;
    if (!parse_token_csv(token_arg, &tokens, &n_tokens)) {
        printf("Invalid token_id_or_csv: %s\n", token_arg);
        return 1;
    }

    // Default behavior: a single token id runs as a batch of 32 identical tokens
    // to match the reference graph shapes. CSV input keeps its explicit batch size.
    if (n_tokens == 1 && strchr(token_arg, ',') == NULL) {
        int target_n_tokens = 32;

        const char *env_n_tokens = getenv("GGML_DEBUG_NTOKENS");
        if (env_n_tokens != NULL) {
            int v = atoi(env_n_tokens);
            if (v > 0) {
                target_n_tokens = v;
            }
        }

        if (target_n_tokens > 1) {
            int *grown = (int *)realloc(tokens, (size_t)target_n_tokens * sizeof(int));
            if (grown == NULL) {
                printf("Failed to expand single token to batch size %d\n", target_n_tokens);
                free(tokens);
                return 1;
            }

            tokens = grown;
            for (int i = 1; i < target_n_tokens; ++i) {
                tokens[i] = tokens[0];
            }
            n_tokens = target_n_tokens;
        }
    }

    struct model model = {0};

    printf("Loading model from %s...\n", path);

    if (!model_load(path, &model)) {
        printf("Failed to load model\n");
        free(tokens);
        return 1;
    }

    if (!n_past_from_cli) {
        n_past = model.hparams.n_ctx - n_tokens;
        if (n_past < 0) {
            n_past = 0;
        }
    }

    printf("Batch run config: n_tokens=%d n_past=%d\n", n_tokens, n_past);

    printf("Model loaded successfully!\n");

    if (!run_batch_inference(&model, tokens, n_tokens, n_past)) {
        printf("Inference failed\n");
        free(tokens);
        gguf_free(model.ctxgguf);
        return 1;
    }

    free(tokens);
    gguf_free(model.ctxgguf);
    return 0;
}
