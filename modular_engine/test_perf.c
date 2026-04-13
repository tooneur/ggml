#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf.h"
#include "parser.h"

typedef struct {
	const char *name;
	int n_nodes;
	int use_topology_tree;
} bench_mode_t;

typedef struct {
	double time_min_s;
	double time_max_s;
	double time_avg_s;
	double tokens_per_sec;
	double comm_bytes_avg;
	double comm_bytes_max;
	double comm_bytes_total;
	double comm_bytes_per_token;
	double comm_estimated_net_s;
	int comm_cut_edges_avg;
	int runs;
} bench_result_t;

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

static double monotonic_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double env_get_double_or_default(const char *name, double default_value) {
	const char *v = getenv(name);
	if (v == NULL || *v == '\0') {
		return default_value;
	}

	char *end = NULL;
	double parsed = strtod(v, &end);
	if (end == v || *end != '\0' || parsed <= 0.0) {
		return default_value;
	}

	return parsed;
}

static void configure_mode_env(const bench_mode_t *mode) {
	char nodes_buf[32];
	snprintf(nodes_buf, sizeof(nodes_buf), "%d", mode->n_nodes);

	setenv("GGML_DISTRIB_NODES", nodes_buf, 1);
	setenv("GGML_DISTRIB_TOPOLOGY_TREE", mode->use_topology_tree ? "1" : "0", 1);
	setenv("GGML_DISTRIB_DOT_EXPORT", mode->n_nodes > 1 ? "1" : "0", 1);
	setenv("GGML_DISTRIB_VERBOSE", "0", 1);
}

static bool parse_partition_dot_metrics(const char *dot_path, double *out_bytes, int *out_cut_edges) {
	FILE *f = fopen(dot_path, "r");
	if (f == NULL) {
		return false;
	}

	double total_bytes = 0.0;
	int cut_edges = 0;
	char line[1024];

	while (fgets(line, sizeof(line), f) != NULL) {
		if (strstr(line, "style=\"bold\"") == NULL) {
			continue;
		}

		char *label = strstr(line, "label=\"");
		if (label == NULL) {
			continue;
		}

		label += 7;
		char *endptr = NULL;
		double w = strtod(label, &endptr);
		if (endptr == label || w < 0.0) {
			continue;
		}

		total_bytes += w;
		cut_edges += 1;
	}

	fclose(f);
	*out_bytes = total_bytes;
	*out_cut_edges = cut_edges;
	return true;
}

static bool run_mode_benchmark(
	struct model *model,
	const int *tokens,
	int n_tokens,
	int n_past,
	int n_iters,
	const bench_mode_t *mode,
	bench_result_t *result) {
	double sum_time = 0.0;
	double sum_comm_bytes = 0.0;
	double max_comm_bytes = 0.0;
	int sum_cut_edges = 0;

	result->time_min_s = DBL_MAX;
	result->time_max_s = 0.0;
	result->runs = 0;

	for (int i = 0; i < n_iters; ++i) {
		configure_mode_env(mode);

		const double t0 = monotonic_seconds();
		const bool ok = run_batch_inference(model, tokens, n_tokens, n_past);
		const double t1 = monotonic_seconds();
		if (!ok) {
			return false;
		}

		const double dt = t1 - t0;
		sum_time += dt;
		if (dt < result->time_min_s) {
			result->time_min_s = dt;
		}
		if (dt > result->time_max_s) {
			result->time_max_s = dt;
		}

		if (mode->n_nodes > 1) {
			double comm_bytes = 0.0;
			int cut_edges = 0;
			if (parse_partition_dot_metrics("partition-plan.dot", &comm_bytes, &cut_edges)) {
				sum_comm_bytes += comm_bytes;
				sum_cut_edges += cut_edges;
				if (comm_bytes > max_comm_bytes) {
					max_comm_bytes = comm_bytes;
				}
			}
		}

		result->runs += 1;
	}

	result->time_avg_s = sum_time / (double)n_iters;
	result->tokens_per_sec = result->time_avg_s > 0.0 ? ((double)n_tokens / result->time_avg_s) : 0.0;

	if (mode->n_nodes > 1) {
		result->comm_bytes_avg = sum_comm_bytes / (double)n_iters;
		result->comm_bytes_max = max_comm_bytes;
		result->comm_bytes_total = sum_comm_bytes;
		result->comm_bytes_per_token = n_tokens > 0 ? (result->comm_bytes_avg / (double)n_tokens) : 0.0;
		result->comm_cut_edges_avg = (int)((double)sum_cut_edges / (double)n_iters + 0.5);

		const double bw_gbps = env_get_double_or_default("GGML_DISTRIB_ESTIMATED_BW_GBPS", 10.0);
		const double bw_bytes_per_s = bw_gbps * 1e9;
		result->comm_estimated_net_s = bw_bytes_per_s > 0.0 ? (result->comm_bytes_avg / bw_bytes_per_s) : 0.0;
	} else {
		result->comm_bytes_avg = 0.0;
		result->comm_bytes_max = 0.0;
		result->comm_bytes_total = 0.0;
		result->comm_bytes_per_token = 0.0;
		result->comm_estimated_net_s = 0.0;
		result->comm_cut_edges_avg = 0;
	}

	return true;
}

static void print_result_row(const bench_mode_t *mode, const bench_result_t *r, int n_tokens) {
	printf("%-16s | nodes=%d tree=%d | avg=%.4fs min=%.4fs max=%.4fs | tok/s=%.1f | comm_avg=%.0fB comm_max=%.0fB cut_edges~%d\n",
		   mode->name,
		   mode->n_nodes,
		   mode->use_topology_tree,
		   r->time_avg_s,
		   r->time_min_s,
		   r->time_max_s,
		   r->tokens_per_sec,
		   r->comm_bytes_avg,
		   r->comm_bytes_max,
		   r->comm_cut_edges_avg);

	if (mode->n_nodes > 1) {
		const double avg_mb = r->comm_bytes_avg / (1024.0 * 1024.0);
		const double total_mb = r->comm_bytes_total / (1024.0 * 1024.0);
		const double per_token_kb = r->comm_bytes_per_token / 1024.0;
		const int transfer_pct = r->time_avg_s > 0.0 ? (int)(100.0 * r->comm_estimated_net_s / r->time_avg_s + 0.5) : 0;

		printf("  share_estimate: avg=%.2f MB/run, total=%.2f MB, %.2f KB/token (n_tokens=%d)\n",
			   avg_mb,
			   total_mb,
			   per_token_kb,
			   n_tokens);
		printf("  net_lower_bound: %.3f ms/run at GGML_DISTRIB_ESTIMATED_BW_GBPS (default 10), ~%d%% of avg runtime\n",
			   r->comm_estimated_net_s * 1000.0,
			   transfer_pct);
	}
}

int main(int argc, char **argv) {
	if (argc < 3) {
		printf("Usage: %s model.gguf token_id_or_csv [n_past] [iters] [n_nodes]\n", argv[0]);
		printf("Example: %s models/gpt-2-117M/ggml-model-f32.gguf 1,2,3,4 0 5 4\n", argv[0]);
		return 1;
	}

	const char *path = argv[1];
	const char *token_arg = argv[2];

	int n_past = 0;
	int n_iters = 5;
	int n_nodes = 4;

	if (argc >= 4 && !parse_non_negative_int(argv[3], &n_past)) {
		printf("Invalid n_past: %s\n", argv[3]);
		return 1;
	}
	if (argc >= 5 && (!parse_non_negative_int(argv[4], &n_iters) || n_iters <= 0)) {
		printf("Invalid iters: %s\n", argv[4]);
		return 1;
	}
	if (argc >= 6 && (!parse_non_negative_int(argv[5], &n_nodes) || n_nodes <= 1)) {
		printf("Invalid n_nodes (must be >= 2): %s\n", argv[5]);
		return 1;
	}

	int *tokens = NULL;
	int n_tokens = 0;
	if (!parse_token_csv(token_arg, &tokens, &n_tokens)) {
		printf("Invalid token_id_or_csv: %s\n", token_arg);
		return 1;
	}

	struct model model = {0};
	printf("Loading model: %s\n", path);
	if (!model_load(path, &model)) {
		printf("Failed to load model\n");
		free(tokens);
		return 1;
	}

	bench_mode_t modes[3] = {
		{.name = "baseline", .n_nodes = 1, .use_topology_tree = 0},
		{.name = "mincut", .n_nodes = n_nodes, .use_topology_tree = 0},
		{.name = "topology-tree", .n_nodes = n_nodes, .use_topology_tree = 1},
	};

	printf("Benchmark config: n_tokens=%d n_past=%d iterations=%d n_nodes=%d\n", n_tokens, n_past, n_iters, n_nodes);
	printf("Warmup run (baseline) ...\n");

	configure_mode_env(&modes[0]);
	if (!run_batch_inference(&model, tokens, n_tokens, n_past)) {
		printf("Warmup failed\n");
		free(tokens);
		gguf_free(model.ctxgguf);
		free(model.model_name);
		return 1;
	}

	printf("\n=== Performance Summary ===\n");
	bench_result_t results[3] = {0};
	for (int i = 0; i < 3; ++i) {
		if (!run_mode_benchmark(&model, tokens, n_tokens, n_past, n_iters, &modes[i], &results[i])) {
			printf("Benchmark failed for mode: %s\n", modes[i].name);
			free(tokens);
			gguf_free(model.ctxgguf);
			free(model.model_name);
			return 1;
		}
		print_result_row(&modes[i], &results[i], n_tokens);
	}

	const double base = results[0].time_avg_s;
	if (base > 0.0) {
		const double mincut_speedup = base / results[1].time_avg_s;
		const double topo_speedup = base / results[2].time_avg_s;
		printf("\nRelative speedup vs baseline: mincut=%.3fx topology-tree=%.3fx\n", mincut_speedup, topo_speedup);
	}

	printf("Note: comm bytes are estimated from inter-partition edges in partition-plan.dot.\n");
	printf("Note: set GGML_DISTRIB_ESTIMATED_BW_GBPS to model network bandwidth (example: 25 for 25 GB/s).\n");

	free(tokens);
	gguf_free(model.ctxgguf);
	free(model.model_name);
	return 0;
}
