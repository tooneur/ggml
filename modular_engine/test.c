#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include "ggml.h"
#include "gguf.h"
#include "parser.h"

#define TEST_PATH_BUF_SIZE 4096

static bool parse_int_arg(const char *text, int *value) {
    if (text == NULL || *text == '\0') {
        return false;
    }

    errno = 0;
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }

    *value = (int) parsed;
    return true;
}

// Helper function to read logits from file
bool read_logits_file(const char *filename, float **logits, int *count) {
    const char *paths[3] = { filename, NULL, NULL };
    char build_path[TEST_PATH_BUF_SIZE] = {0};
    char parent_build_path[TEST_PATH_BUF_SIZE] = {0};

    
    if (strchr(filename, '/') == NULL) {
        snprintf(build_path, sizeof(build_path), "build/%s", filename);
        snprintf(parent_build_path, sizeof(parent_build_path), "../build/%s", filename);
        paths[1] = build_path;
        paths[2] = parent_build_path;
    }

    FILE *f = NULL;
    const char *used_path = NULL;
    for (int i = 0; i < 3; ++i) {
        if (paths[i] == NULL) {
            continue;
        }
        f = fopen(paths[i], "rb");
        if (f != NULL) {
            used_path = paths[i];
            break;
        }
    }

    if (f == NULL) {
        char cwd[TEST_PATH_BUF_SIZE] = {0};
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            fprintf(stderr, "Failed to open file: %s (cwd: %s)\n", filename, cwd);
        } else {
            fprintf(stderr, "Failed to open file: %s\n", filename);
        }
        return false;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || (file_size % (long) sizeof(float)) != 0) {
        fprintf(stderr, "Invalid logits file size for %s\n", used_path);
        fclose(f);
        return false;
    }

    *count = file_size / sizeof(float);
    *logits = (float *)malloc(file_size);
    if (*logits == NULL) {
        fprintf(stderr, "Failed to allocate memory for logits\n");
        fclose(f);
        return false;
    }

    if (fread(*logits, sizeof(float), *count, f) != (size_t)*count) {
        fprintf(stderr, "Failed to read logits from file\n");
        free(*logits);
        *logits = NULL;
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// Helper function to compute stable softmax
void compute_softmax(float *logits, float *probs, int count) {
    // Find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < count; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // Compute exp(logits - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < count; i++) {
        probs[i] /= sum;
    }
}

// Helper function to compare probabilities with tolerance
bool compare_logits(float *logits1, float *logits2, int count, float tolerance) {
    // Compute softmax (probabilities) for both logits
    float *probs1 = (float *)malloc(count * sizeof(float));
    float *probs2 = (float *)malloc(count * sizeof(float));

    if (probs1 == NULL || probs2 == NULL) {
        fprintf(stderr, "Failed to allocate memory for probabilities\n");
        free(probs1);
        free(probs2);
        return false;
    }

    compute_softmax(logits1, probs1, count);
    compute_softmax(logits2, probs2, count);

    int mismatch_count = 0;
    float max_diff = 0.0f;
    int max_diff_index = -1;

    for (int i = 0; i < count; i++) {
        float diff = probs1[i] - probs2[i];
        if (diff < 0) diff = -diff;

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_index = i;
        }

        if (diff > tolerance) {
            mismatch_count++;
            if (mismatch_count <= 15) {  // Only print first 10 mismatches to avoid spam
                printf("Mismatch at index %d: prob %f vs %f (diff: %f)\n",
                       i, probs1[i], probs2[i], diff);
            }
        }
    }

    if (mismatch_count > 0) {
        printf("Total mismatches: %d / %d (tolerance: %e)\n", mismatch_count, count, tolerance);
        printf("Max diff at index %d: prob %f vs %f (diff: %f)\n",
               max_diff_index,
               probs1[max_diff_index],
               probs2[max_diff_index],
               max_diff);
        free(probs1);
        free(probs2);
        return false;
    }

    printf("All probabilities match (max diff: %e)\n", max_diff);
    free(probs1);
    free(probs2);
    return true;
}

// Structure to hold top-k token info
typedef struct {
    int token_id;
    float probability;
} top_k_token_t;

// Comparison function for qsort (descending order by probability)
int compare_topk_tokens(const void *a, const void *b) {
    float prob_a = ((top_k_token_t *)a)->probability;
    float prob_b = ((top_k_token_t *)b)->probability;
    if (prob_a > prob_b) return -1;
    if (prob_a < prob_b) return 1;
    return 0;
}

// Helper function to get top-k tokens
void get_topk_tokens(float *logits, int count, int k, top_k_token_t *topk) {
    // Compute probabilities
    float *probs = (float *)malloc(count * sizeof(float));
    compute_softmax(logits, probs, count);

    // Create array of token info
    top_k_token_t *tokens = (top_k_token_t *)malloc(count * sizeof(top_k_token_t));
    for (int i = 0; i < count; i++) {
        tokens[i].token_id = i;
        tokens[i].probability = probs[i];
    }

    // Sort by probability (descending)
    qsort(tokens, count, sizeof(top_k_token_t), compare_topk_tokens);

    // Copy top-k
    for (int i = 0; i < k; i++) {
        topk[i] = tokens[i];
    }

    free(tokens);
    free(probs);
}

// Test 4: Top-K consistency
bool test_topk_consistency(const char *model_path, const char *reference_logits_path, int token_id, int k) {
    printf("\n=== Test 4: Top-K Consistency ===\n");
    struct model model = {0};

    if (!model_load(model_path, &model)) {
        printf("FAIL: Model failed to load\n");
        return false;
    }

    if (!run_single_token_inference(&model, token_id)) {
        printf("FAIL: Inference failed\n");
        return false;
    }

    float *generated_logits = NULL;
    int generated_count = 0;
    if (!read_logits_file("logits-custom.bin", &generated_logits, &generated_count)) {
        printf("FAIL: Could not read generated logits\n");
        return false;
    }

    float *reference_logits = NULL;
    int reference_count = 0;
    if (!read_logits_file(reference_logits_path, &reference_logits, &reference_count)) {
        printf("FAIL: Could not read reference logits\n");
        free(generated_logits);
        return false;
    }

    if (generated_count != reference_count) {
        printf("FAIL: Logits count mismatch: %d vs %d\n", generated_count, reference_count);
        free(generated_logits);
        free(reference_logits);
        return false;
    }

    // Get top-k for both
    top_k_token_t *topk_generated = (top_k_token_t *)malloc(k * sizeof(top_k_token_t));
    top_k_token_t *topk_reference = (top_k_token_t *)malloc(k * sizeof(top_k_token_t));

    get_topk_tokens(generated_logits, generated_count, k, topk_generated);
    get_topk_tokens(reference_logits, reference_count, k, topk_reference);

    printf("Generated top-%d tokens:\n", k);
    for (int i = 0; i < k; i++) {
        printf("  %d. Token %d (prob: %.6f)\n", i+1, topk_generated[i].token_id, topk_generated[i].probability);
    }

    printf("Reference top-%d tokens:\n", k);
    for (int i = 0; i < k; i++) {
        printf("  %d. Token %d (prob: %.6f)\n", i+1, topk_reference[i].token_id, topk_reference[i].probability);
    }

    // Check if top-1 is the same
    bool top1_match = (topk_generated[0].token_id == topk_reference[0].token_id);
    if (!top1_match) {
        printf("WARNING: Top-1 prediction differs! Generated: %d, Reference: %d\n",
               topk_generated[0].token_id, topk_reference[0].token_id);
    } else {
        printf("✓ Top-1 prediction matches\n");
    }

    // Check if top-k set is the same
    bool topk_match = true;
    for (int i = 0; i < k; i++) {
        bool found = false;
        for (int j = 0; j < k; j++) {
            if (topk_generated[i].token_id == topk_reference[j].token_id) {
                found = true;
                break;
            }
        }
        if (!found) {
            topk_match = false;
            printf("WARNING: Token %d not in reference top-%d\n", topk_generated[i].token_id, k);
        }
    }

    if (topk_match && top1_match) {
        printf("PASS: Top-%d predictions match perfectly\n", k);
        free(topk_generated);
        free(topk_reference);
        free(generated_logits);
        free(reference_logits);
        return true;
    } else if (top1_match) {
        printf("PASS: Top-1 matches (differences in top-%d diversity)\n", k);
        free(topk_generated);
        free(topk_reference);
        free(generated_logits);
        free(reference_logits);
        return true;
    } else {
        printf("FAIL: Top-1 prediction differs - this WILL impact inference\n");
        free(topk_generated);
        free(topk_reference);
        free(generated_logits);
        free(reference_logits);
        return false;
    }
}

// Test 1: Model loading
bool test_model_loading(const char *model_path) {
    printf("\n=== Test 1: Model Loading ===\n");
    struct model model = {0};

    if (!model_load(model_path, &model)) {
        printf("FAIL: Model failed to load\n");
        return false;
    }

    if (model.model_name == NULL) {
        printf("FAIL: Model name is NULL\n");
        return false;
    }

    if (model.hparams.n_vocab <= 0 || model.hparams.n_embd <= 0) {
        printf("FAIL: Invalid hyperparameters\n");
        return false;
    }

    printf("PASS: Model loaded successfully\n");
    printf("  Model name: %s\n", model.model_name);
    printf("  n_vocab: %d, n_embd: %d, n_layer: %d\n",
           model.hparams.n_vocab, model.hparams.n_embd, model.hparams.n_layer);

    return true;
}

bool test_single_token_inference(const char *model_path, int token_id) {
    printf("\n=== Test 2: Single Token Inference ===\n");
    struct model model = {0};

    if (!model_load(model_path, &model)) {
        printf("FAIL: Model failed to load\n");
        return false;
    }

    if (!run_single_token_inference(&model, token_id)) {
        printf("FAIL: Inference failed for token %d\n", token_id);
        return false;
    }

    printf("PASS: Inference completed for token %d\n", token_id);
    return true;
}

// Test 3: Logits output consistency
bool test_logits_consistency(const char *model_path, const char *reference_logits_path, int token_id) {
    printf("\n=== Test 3: Logits Consistency (vs Reference) ===\n");
    struct model model = {0};

    if (!model_load(model_path, &model)) {
        printf("FAIL: Model failed to load\n");
        return false;
    }

    if (!run_single_token_inference(&model, token_id)) {
        printf("FAIL: Inference failed\n");
        return false;
    }

    float *generated_logits = NULL;
    int generated_count = 0;
    if (!read_logits_file("logits-custom.bin", &generated_logits, &generated_count)) {
        printf("FAIL: Could not read generated logits\n");
        return false;
    }

    float *reference_logits = NULL;
    int reference_count = 0;
    if (!read_logits_file(reference_logits_path, &reference_logits, &reference_count)) {
        printf("FAIL: Could not read reference logits\n");
        free(generated_logits);
        return false;
    }

    if (generated_count != reference_count) {
        printf("FAIL: Logits count mismatch: %d vs %d\n", generated_count, reference_count);
        free(generated_logits);
        free(reference_logits);
        return false;
    }

    float tolerance = 1e-3f;
    if (!compare_logits(generated_logits, reference_logits, generated_count, tolerance)) {
        printf("FAIL: Logits values differ beyond tolerance\n");
        free(generated_logits);
        free(reference_logits);
        return false;
    }
    

    printf("PASS: Logits match reference implementation (tolerance: %e)\n", tolerance);
    free(generated_logits);
    free(reference_logits);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s model.gguf [token_id] [reference_logits.bin] [top_k]\n", argv[0]);
        printf("   or: %s model.gguf [reference_logits.bin] [token_id] [top_k]\n", argv[0]);
        printf("\nOptions:\n");
        printf("  top_k: Number of top predictions to compare (default: 5)\n");
        return 1;
    }

    const char *model_path = argv[1];
    const char *reference_logits = "logits-reference.bin";
    int token_id = 1;
    int top_k = 5;

    if (argc >= 3) {
        int parsed = 0;
        if (parse_int_arg(argv[2], &parsed)) {
            token_id = parsed;
        } else {
            reference_logits = argv[2];
        }
    }

    if (argc >= 4) {
        int parsed = 0;
        if (parse_int_arg(argv[3], &parsed)) {
            if (token_id == 1) {  // argv[2] was not a number, so argv[3] is token_id
                token_id = parsed;
            } else {  // argv[2] was token_id, so argv[3] is top_k
                top_k = parsed;
            }
        } else {
            reference_logits = argv[3];
        }
    }

    if (argc >= 5) {
        int parsed = 0;
        if (parse_int_arg(argv[4], &parsed)) {
            top_k = parsed;
        }
    }

    printf("Running tests for parser.c...\n");
    printf("Model: %s | Token ID: %d | Reference: %s | Top-K: %d\n\n", 
           model_path, token_id, reference_logits, top_k);

    bool pass1 = test_model_loading(model_path);
    bool pass2 = test_single_token_inference(model_path, token_id);
    bool pass3 = test_logits_consistency(model_path, reference_logits, token_id);
    bool pass4 = test_topk_consistency(model_path, reference_logits, token_id, top_k);

    printf("\n=== Test Summary ===\n");
    printf("Model Loading: %s\n", pass1 ? "PASS" : "FAIL");
    printf("Single Token Inference: %s\n", pass2 ? "PASS" : "FAIL");
    printf("Logits Consistency: %s\n", pass3 ? "PASS" : "FAIL");
    printf("Top-K Consistency (k=%d): %s\n", top_k, pass4 ? "PASS" : "FAIL");

    return (pass1 && pass2 && pass3 && pass4) ? 0 : 1;
}