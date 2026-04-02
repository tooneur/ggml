#define _GNU_SOURCE
#include "distrib.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Portable qsort_r wrapper */
#ifdef __GLIBC__
#define HAVE_QSORT_R 1
#else
#define HAVE_QSORT_R 0
#endif

typedef struct {
	int *out_begin;
	int *out_edges;
	int *in_begin;
	int *in_edges;
} adjacency_index_t;

typedef struct {
	const double *cost;
} sort_ctx_t;

static sort_ctx_t sort_context;

static int cmp_vertex_desc_cost(const void *a, const void *b, void *ctx) {
	const double *cost = (const double *)ctx;
	const int ia = *(const int *)a;
	const int ib = *(const int *)b;
	if (cost[ia] < cost[ib]) {
		return 1;
	}
	if (cost[ia] > cost[ib]) {
		return -1;
	}
	return ia - ib;
}

static int cmp_vertex_desc_cost_portable(const void *a, const void *b) {
	const sort_ctx_t *ctx = &sort_context;
	const double *cost = ctx->cost;
	const int ia = *(const int *)a;
	const int ib = *(const int *)b;
	if (cost[ia] < cost[ib]) {
		return 1;
	}
	if (cost[ia] > cost[ib]) {
		return -1;
	}
	return ia - ib;
}

static sort_ctx_t sort_context;

static bool validate_inputs(
	const dist_graph_t *graph,
	const dist_partition_config_t *cfg,
	const dist_partition_result_t *res) {
	if (graph == NULL || cfg == NULL || res == NULL) {
		return false;
	}
	if (graph->n_vertices <= 0 || graph->n_edges < 0 || cfg->n_partitions <= 0) {
		return false;
	}
	if (graph->vertex_compute_cost == NULL || (graph->n_edges > 0 && graph->edges == NULL)) {
		return false;
	}
	if (res->vertex_to_partition == NULL || res->partition_loads == NULL) {
		return false;
	}
	if (cfg->lambda_comm < 0.0 || cfg->lambda_balance < 0.0) {
		return false;
	}
	if (cfg->max_refine_iters < 0) {
		return false;
	}

	for (int i = 0; i < graph->n_vertices; ++i) {
		if (!isfinite(graph->vertex_compute_cost[i]) || graph->vertex_compute_cost[i] < 0.0) {
			return false;
		}
	}

	for (int e = 0; e < graph->n_edges; ++e) {
		if (graph->edges[e].src < 0 || graph->edges[e].src >= graph->n_vertices) {
			return false;
		}
		if (graph->edges[e].dst < 0 || graph->edges[e].dst >= graph->n_vertices) {
			return false;
		}
		if (!isfinite(graph->edges[e].weight) || graph->edges[e].weight < 0.0) {
			return false;
		}
	}

	return true;
}

static double load_penalty_single(double load, double target) {
	const double denom = target > 1e-12 ? target : 1.0;
	const double x = (load - target) / denom;
	return x * x;
}

static bool build_targets(const dist_graph_t *graph, const dist_partition_config_t *cfg, double *targets) {
	if (cfg->target_partition_load != NULL) {
		for (int p = 0; p < cfg->n_partitions; ++p) {
			if (!isfinite(cfg->target_partition_load[p]) || cfg->target_partition_load[p] <= 0.0) {
				return false;
			}
			targets[p] = cfg->target_partition_load[p];
		}
		return true;
	}

	double total = 0.0;
	for (int v = 0; v < graph->n_vertices; ++v) {
		total += graph->vertex_compute_cost[v];
	}
	const double equal = total > 0.0 ? (total / (double)cfg->n_partitions) : 1.0;
	for (int p = 0; p < cfg->n_partitions; ++p) {
		targets[p] = equal;
	}
	return true;
}

static bool build_adjacency(const dist_graph_t *graph, adjacency_index_t *adj) {
	const int n = graph->n_vertices;
	const int m = graph->n_edges;

	int *out_deg = (int *)calloc((size_t)n, sizeof(int));
	int *in_deg = (int *)calloc((size_t)n, sizeof(int));
	if (out_deg == NULL || in_deg == NULL) {
		free(out_deg);
		free(in_deg);
		return false;
	}

	for (int e = 0; e < m; ++e) {
		out_deg[graph->edges[e].src]++;
		in_deg[graph->edges[e].dst]++;
	}

	adj->out_begin = (int *)malloc((size_t)(n + 1) * sizeof(int));
	adj->in_begin = (int *)malloc((size_t)(n + 1) * sizeof(int));
	adj->out_edges = (int *)malloc((size_t)m * sizeof(int));
	adj->in_edges = (int *)malloc((size_t)m * sizeof(int));
	if (adj->out_begin == NULL || adj->in_begin == NULL || adj->out_edges == NULL || adj->in_edges == NULL) {
		free(out_deg);
		free(in_deg);
		free(adj->out_begin);
		free(adj->in_begin);
		free(adj->out_edges);
		free(adj->in_edges);
		memset(adj, 0, sizeof(*adj));
		return false;
	}

	adj->out_begin[0] = 0;
	adj->in_begin[0] = 0;
	for (int v = 0; v < n; ++v) {
		adj->out_begin[v + 1] = adj->out_begin[v] + out_deg[v];
		adj->in_begin[v + 1] = adj->in_begin[v] + in_deg[v];
	}

	int *out_write = (int *)malloc((size_t)n * sizeof(int));
	int *in_write = (int *)malloc((size_t)n * sizeof(int));
	if (out_write == NULL || in_write == NULL) {
		free(out_deg);
		free(in_deg);
		free(out_write);
		free(in_write);
		free(adj->out_begin);
		free(adj->in_begin);
		free(adj->out_edges);
		free(adj->in_edges);
		memset(adj, 0, sizeof(*adj));
		return false;
	}

	memcpy(out_write, adj->out_begin, (size_t)n * sizeof(int));
	memcpy(in_write, adj->in_begin, (size_t)n * sizeof(int));

	for (int e = 0; e < m; ++e) {
		const int src = graph->edges[e].src;
		const int dst = graph->edges[e].dst;
		adj->out_edges[out_write[src]++] = e;
		adj->in_edges[in_write[dst]++] = e;
	}

	free(out_write);
	free(in_write);
	free(out_deg);
	free(in_deg);
	return true;
}

static void free_adjacency(adjacency_index_t *adj) {
	free(adj->out_begin);
	free(adj->out_edges);
	free(adj->in_begin);
	free(adj->in_edges);
	memset(adj, 0, sizeof(*adj));
}

static double compute_cut_cost(
	const dist_graph_t *graph,
	const int *part_of,
	int edge_skip,
	int moved_vertex,
	int moved_to_part) {
	double cut = 0.0;
	for (int e = 0; e < graph->n_edges; ++e) {
		if (e == edge_skip) {
			continue;
		}
		int ps = part_of[graph->edges[e].src];
		int pd = part_of[graph->edges[e].dst];

		if (graph->edges[e].src == moved_vertex) {
			ps = moved_to_part;
		}
		if (graph->edges[e].dst == moved_vertex) {
			pd = moved_to_part;
		}

		if (ps != pd) {
			cut += graph->edges[e].weight;
		}
	}
	return cut;
}

static double compute_balance_penalty(
	const double *loads,
	const double *targets,
	int n_parts,
	int moved_from,
	int moved_to,
	double move_cost) {
	double penalty = 0.0;
	for (int p = 0; p < n_parts; ++p) {
		double lp = loads[p];
		if (p == moved_from) {
			lp -= move_cost;
		}
		if (p == moved_to) {
			lp += move_cost;
		}
		penalty += load_penalty_single(lp, targets[p]);
	}
	return penalty;
}

static bool capacity_ok(
	const dist_partition_config_t *cfg,
	const double *targets,
	const double *loads,
	int from,
	int to,
	double v_cost) {
	if (cfg->hard_cap_ratio < 0.0) {
		return true;
	}
	const double max_to = targets[to] * (1.0 + cfg->hard_cap_ratio);
	const double next_to = loads[to] + v_cost;
	const double next_from = loads[from] - v_cost;
	if (next_to > max_to + 1e-12) {
		return false;
	}
	if (next_from < -1e-12) {
		return false;
	}
	return true;
}

static void greedy_initialize(
	const dist_graph_t *graph,
	const dist_partition_config_t *cfg,
	const double *targets,
	const adjacency_index_t *adj,
	int *part_of,
	double *loads) {
	const int n = graph->n_vertices;
	const int k = cfg->n_partitions;

	int *order = (int *)malloc((size_t)n * sizeof(int));
	if (order == NULL) {
		// Fallback: round-robin assignment.
		for (int p = 0; p < k; ++p) {
			loads[p] = 0.0;
		}
		for (int v = 0; v < n; ++v) {
			int p = v % k;
			part_of[v] = p;
			loads[p] += graph->vertex_compute_cost[v];
		}
		return;
	}

	for (int v = 0; v < n; ++v) {
		order[v] = v;
	}

#if defined(__GLIBC__)
	qsort_r(order, (size_t)n, sizeof(int), cmp_vertex_desc_cost, (void *)graph->vertex_compute_cost);
#else
	// Portable fallback for libc variants where qsort_r signature differs.
	for (int i = 0; i < n - 1; ++i) {
		for (int j = i + 1; j < n; ++j) {
			if (graph->vertex_compute_cost[order[j]] > graph->vertex_compute_cost[order[i]]) {
				int t = order[i];
				order[i] = order[j];
				order[j] = t;
			}
		}
	}
#endif

	for (int p = 0; p < k; ++p) {
		loads[p] = 0.0;
	}
	for (int v = 0; v < n; ++v) {
		part_of[v] = -1;
	}

	for (int i = 0; i < n; ++i) {
		const int v = order[i];
		const double v_cost = graph->vertex_compute_cost[v];

		int best_p = 0;
		double best_score = DBL_MAX;

		for (int p = 0; p < k; ++p) {
			if (cfg->hard_cap_ratio >= 0.0) {
				const double max_p = targets[p] * (1.0 + cfg->hard_cap_ratio);
				if (loads[p] + v_cost > max_p + 1e-12) {
					continue;
				}
			}

			double comm_penalty = 0.0;

			for (int oi = adj->out_begin[v]; oi < adj->out_begin[v + 1]; ++oi) {
				const int e = adj->out_edges[oi];
				const int u = graph->edges[e].dst;
				if (part_of[u] >= 0 && part_of[u] != p) {
					comm_penalty += graph->edges[e].weight;
				}
			}
			for (int ii = adj->in_begin[v]; ii < adj->in_begin[v + 1]; ++ii) {
				const int e = adj->in_edges[ii];
				const int u = graph->edges[e].src;
				if (part_of[u] >= 0 && part_of[u] != p) {
					comm_penalty += graph->edges[e].weight;
				}
			}

			const double load_ratio = (loads[p] + v_cost) / (targets[p] > 1e-12 ? targets[p] : 1.0);
			const double score = cfg->lambda_balance * load_ratio + cfg->lambda_comm * comm_penalty;
			if (score < best_score) {
				best_score = score;
				best_p = p;
			}
		}

		part_of[v] = best_p;
		loads[best_p] += v_cost;
	}

	free(order);
}

static double delta_cut_if_move(
	const dist_graph_t *graph,
	const adjacency_index_t *adj,
	const int *part_of,
	int v,
	int from,
	int to) {
	double delta = 0.0;

	for (int oi = adj->out_begin[v]; oi < adj->out_begin[v + 1]; ++oi) {
		const int e = adj->out_edges[oi];
		const int u = graph->edges[e].dst;
		const int pu = part_of[u];
		const double w = graph->edges[e].weight;

		const double old_cross = (from != pu) ? w : 0.0;
		const double new_cross = (to != pu) ? w : 0.0;
		delta += new_cross - old_cross;
	}

	for (int ii = adj->in_begin[v]; ii < adj->in_begin[v + 1]; ++ii) {
		const int e = adj->in_edges[ii];
		const int u = graph->edges[e].src;
		const int pu = part_of[u];
		const double w = graph->edges[e].weight;

		const double old_cross = (pu != from) ? w : 0.0;
		const double new_cross = (pu != to) ? w : 0.0;
		delta += new_cross - old_cross;
	}

	return delta;
}

bool dist_partition_graph_balanced_min_cut(
	const dist_graph_t *graph,
	const dist_partition_config_t *config,
	dist_partition_result_t *result) {
	if (!validate_inputs(graph, config, result)) {
		return false;
	}

	if (config->n_partitions > graph->n_vertices) {
		// Allowed, but some partitions will remain empty.
	}

	const int n = graph->n_vertices;
	const int k = config->n_partitions;

	double *targets = (double *)malloc((size_t)k * sizeof(double));
	if (targets == NULL) {
		return false;
	}
	if (!build_targets(graph, config, targets)) {
		free(targets);
		return false;
	}

	adjacency_index_t adj = {0};
	if (!build_adjacency(graph, &adj)) {
		free(targets);
		return false;
	}

	greedy_initialize(graph, config, targets, &adj, result->vertex_to_partition, result->partition_loads);

	double cut_cost = compute_cut_cost(graph, result->vertex_to_partition, -1, -1, -1);
	double bal_pen = compute_balance_penalty(result->partition_loads, targets, k, -1, -1, 0.0);
	double objective = config->lambda_comm * cut_cost + config->lambda_balance * bal_pen;

	const int max_iters = config->max_refine_iters > 0 ? config->max_refine_iters : 20;
	const double improve_eps = config->min_delta_improvement > 0.0 ? config->min_delta_improvement : 1e-9;

	for (int it = 0; it < max_iters; ++it) {
		bool moved_any = false;

		for (int v = 0; v < n; ++v) {
			const int from = result->vertex_to_partition[v];
			const double v_cost = graph->vertex_compute_cost[v];

			int best_to = from;
			double best_delta_obj = 0.0;

			for (int to = 0; to < k; ++to) {
				if (to == from) {
					continue;
				}
				if (!capacity_ok(config, targets, result->partition_loads, from, to, v_cost)) {
					continue;
				}

				const double d_cut = delta_cut_if_move(graph, &adj, result->vertex_to_partition, v, from, to);

				const double old_pen = load_penalty_single(result->partition_loads[from], targets[from]) +
									   load_penalty_single(result->partition_loads[to], targets[to]);
				const double new_pen = load_penalty_single(result->partition_loads[from] - v_cost, targets[from]) +
									   load_penalty_single(result->partition_loads[to] + v_cost, targets[to]);
				const double d_bal = new_pen - old_pen;

				const double d_obj = config->lambda_comm * d_cut + config->lambda_balance * d_bal;
				if (d_obj < best_delta_obj) {
					best_delta_obj = d_obj;
					best_to = to;
				}
			}

			if (best_to != from && best_delta_obj < -improve_eps) {
				result->vertex_to_partition[v] = best_to;
				result->partition_loads[from] -= v_cost;
				result->partition_loads[best_to] += v_cost;

				// Recompute robustly to avoid drift due to approximation/order effects.
				cut_cost = compute_cut_cost(graph, result->vertex_to_partition, -1, -1, -1);
				bal_pen = compute_balance_penalty(result->partition_loads, targets, k, -1, -1, 0.0);
				objective = config->lambda_comm * cut_cost + config->lambda_balance * bal_pen;

				moved_any = true;
			}
		}

		if (!moved_any) {
			break;
		}
	}

	result->cut_cost = cut_cost;
	result->balance_penalty = bal_pen;
	result->objective = objective;

	free_adjacency(&adj);
	free(targets);
	return true;
}

void dist_compute_partition_comm_stats(
	const dist_graph_t *graph,
	const int *vertex_to_partition,
	int n_partitions,
	double *outgoing_bytes,
	double *incoming_bytes,
	double *internal_bytes)
{
	if (graph == NULL || vertex_to_partition == NULL)
	{
		return;
	}

	for (int p = 0; p < n_partitions; ++p)
	{
		outgoing_bytes[p] = 0.0;
		incoming_bytes[p] = 0.0;
		internal_bytes[p] = 0.0;
	}

	for (int e = 0; e < graph->n_edges; ++e)
	{
		const int src_p = vertex_to_partition[graph->edges[e].src];
		const int dst_p = vertex_to_partition[graph->edges[e].dst];
		const double w = graph->edges[e].weight;

		if (src_p == dst_p)
		{
			internal_bytes[src_p] += w;
		}
		else
		{
			outgoing_bytes[src_p] += w;
			incoming_bytes[dst_p] += w;
		}
	}
}

bool dist_export_partition_to_dot(
	const char *output_path,
	const dist_graph_t *graph,
	const dist_partition_config_t *config,
	const dist_partition_result_t *result,
	const char *const *vertex_labels,
	const char *const *partition_names)
{
	if (output_path == NULL || graph == NULL || config == NULL || result == NULL)
	{
		return false;
	}

	FILE *f = fopen(output_path, "w");
	if (f == NULL)
	{
		return false;
	}

	fprintf(f, "digraph partition_graph {\n");
	fprintf(f, "rankdir=LR;\n");
	fprintf(f, "labelloc=t;\n");
	fprintf(f, "label=\"Graph Partition Visualization\";\n");
	fprintf(f, "fontname=\"Helvetica\";\n");
	fprintf(f, "node [fontname=\"Helvetica\", fontsize=9, shape=box, style=\"rounded,filled\"];\n");
	fprintf(f, "edge [fontname=\"Helvetica\", fontsize=8];\n\n");

	double *outgoing = (double *)malloc((size_t)config->n_partitions * sizeof(double));
	double *incoming = (double *)malloc((size_t)config->n_partitions * sizeof(double));
	double *internal = (double *)malloc((size_t)config->n_partitions * sizeof(double));
	if (outgoing == NULL || incoming == NULL || internal == NULL)
	{
		free(outgoing);
		free(incoming);
		free(internal);
		fclose(f);
		return false;
	}

	dist_compute_partition_comm_stats(graph, result->vertex_to_partition, config->n_partitions,
									   outgoing, incoming, internal);

	fprintf(f, "subgraph cluster_partitions {\n");
	fprintf(f, "label=\"Partitions\";\n");
	fprintf(f, "style=\"filled\";\n");
	fprintf(f, "color=\"#e8e8e8\";\n\n");

	for (int p = 0; p < config->n_partitions; ++p)
	{
		fprintf(f, "subgraph cluster_p%d {\n", p);

		const char *pname = (partition_names != NULL && partition_names[p] != NULL)
							   ? partition_names[p]
							   : "P";
		fprintf(f, "label=\"%s%d\\nout=%.0f in=%.0f int=%.0f\";\n", pname, p, outgoing[p], incoming[p], internal[p]);
		fprintf(f, "style=\"filled,rounded\";\n");

		const char *colors[] = {"#ffffcc", "#ccffcc", "#ccccff", "#ffcccc", "#ccffff", "#ffccff"};
		const char *color = colors[p % (sizeof(colors) / sizeof(colors[0]))];
		fprintf(f, "color=\"%s\";\n", color);
		fprintf(f, "bgcolor=\"%s\";\n\n", color);

		for (int v = 0; v < graph->n_vertices; ++v)
		{
			if (result->vertex_to_partition[v] != p)
			{
				continue;
			}

			const char *vlabel = (vertex_labels != NULL && vertex_labels[v] != NULL)
								  ? vertex_labels[v]
								  : "op";
			const double vcost = graph->vertex_compute_cost[v];

			fprintf(f, "v%d [label=\"%s\\n(%.0f)\", fillcolor=\"white\"];\n", v, vlabel, vcost);
		}

		fprintf(f, "}\n\n");
	}

	fprintf(f, "}\n\n");

	fprintf(f, "subgraph cluster_edges {\n");
	fprintf(f, "label=\"Data Dependencies\";\n");
	fprintf(f, "style=\"invis\";\n\n");

	for (int e = 0; e < graph->n_edges; ++e)
	{
		const int src = graph->edges[e].src;
		const int dst = graph->edges[e].dst;
		const int src_p = result->vertex_to_partition[src];
		const int dst_p = result->vertex_to_partition[dst];
		const double w = graph->edges[e].weight;

		const char *style = "solid";
		const char *color = "#333333";
		int penwidth = 1;

		if (src_p != dst_p)
		{
			style = "bold";
			color = "#cc0000";
			penwidth = 2;
		}

		fprintf(f, "v%d -> v%d [label=\"%.0f\", color=\"%s\", style=\"%s\", penwidth=%d];\n",
				src, dst, w, color, style, penwidth);
	}

	fprintf(f, "}\n\n");

	fprintf(f, "subgraph cluster_stats {\n");
	fprintf(f, "label=\"Partition Statistics\";\n");
	fprintf(f, "style=\"invis\";\n\n");

	fprintf(f, "legend [shape=plaintext, label=<\n");
	fprintf(f, "<TABLE BORDER=\"1\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"5\">\n");
	fprintf(f, "<TR><TD COLSPAN=\"4\"><B>Partition Communication</B></TD></TR>\n");
	fprintf(f, "<TR><TD><B>Partition</B></TD><TD><B>Load</B></TD><TD><B>Cut Bytes</B></TD><TD><B>Utilization</B></TD></TR>\n");

	double total_target = 0.0;
	for (int p = 0; p < config->n_partitions; ++p)
	{
		double target = result->partition_loads[p];
		if (config->target_partition_load != NULL)
		{
			target = config->target_partition_load[p];
		}
		total_target += target;
	}
	if (total_target <= 0.0)
	{
		total_target = 1.0;
	}
	const double avg_target = total_target / (double)config->n_partitions;

	for (int p = 0; p < config->n_partitions; ++p)
	{
		const char *pname = (partition_names != NULL && partition_names[p] != NULL)
							   ? partition_names[p]
							   : "P";
		double util = 0.0;
		if (avg_target > 1e-12)
		{
			util = 100.0 * result->partition_loads[p] / avg_target;
		}
		double cut_bytes = outgoing[p] + incoming[p];

		fprintf(f, "<TR><TD>%s%d</TD><TD>%.0f</TD><TD>%.0f</TD><TD>%.1f%%</TD></TR>\n",
				pname, p, result->partition_loads[p], cut_bytes, util);
	}

	fprintf(f, "<TR><TD COLSPAN=\"4\" ALIGN=\"CENTER\"><B>Objective</B></TD></TR>\n");
	fprintf(f, "<TR><TD COLSPAN=\"4\">cut_cost=%.2f balance_penalty=%.6f obj=%.6f</TD></TR>\n",
			result->cut_cost, result->balance_penalty, result->objective);
	fprintf(f, "</TABLE>\n");
	fprintf(f, ">];\n\n");

	fprintf(f, "}\n}\n");

	fclose(f);

	free(outgoing);
	free(incoming);
	free(internal);

	return true;
}
