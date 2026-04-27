#include "distrib.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kaHIP_interface.h"

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
	if (cfg->n_partitions > graph->n_vertices) {
		return false;
	}
	return true;
}

static int clamp_positive_int(double x) {
	if (!isfinite(x) || x <= 0.0) {
		return 1;
	}
	if (x > (double)std::numeric_limits<int>::max()) {
		return std::numeric_limits<int>::max();
	}
	const long long rounded = (long long)llround(x);
	if (rounded <= 0) {
		return 1;
	}
	if (rounded > (long long)std::numeric_limits<int>::max()) {
		return std::numeric_limits<int>::max();
	}
	return (int)rounded;
}

static double load_penalty_single(double load, double target) {
	const double denom = target > 1e-12 ? target : 1.0;
	const double x = (load - target) / denom;
	return x * x;
}

extern "C" bool dist_partition_graph_kahip(
	const dist_graph_t *graph,
	const dist_partition_config_t *config,
	dist_partition_result_t *result) {
	if (!validate_inputs(graph, config, result)) {
		return false;
	}

	const int n = graph->n_vertices;
	const int k = config->n_partitions;

	std::vector<std::unordered_map<int, int>> undirected((size_t)n);
	undirected.reserve((size_t)n);

	for (int e = 0; e < graph->n_edges; ++e) {
		const int src = graph->edges[e].src;
		const int dst = graph->edges[e].dst;
		if (src < 0 || src >= n || dst < 0 || dst >= n || src == dst) {
			continue;
		}

		const int w = clamp_positive_int(graph->edges[e].weight);
		undirected[(size_t)src][dst] += w;
		undirected[(size_t)dst][src] += w;
	}

	std::vector<kahip_idx> xadj((size_t)n + 1, 0);
	std::vector<kahip_idx> adjncy;
	std::vector<kahip_idx> adjcwgt;

	size_t total_adj = 0;
	for (int v = 0; v < n; ++v) {
		total_adj += undirected[(size_t)v].size();
	}
	adjncy.reserve(total_adj);
	adjcwgt.reserve(total_adj);

	for (int v = 0; v < n; ++v) {
		xadj[(size_t)v] = (kahip_idx)adjncy.size();
		std::vector<std::pair<int, int>> neigh;
		neigh.reserve(undirected[(size_t)v].size());
		for (const auto &kv : undirected[(size_t)v]) {
			neigh.push_back(kv);
		}
		std::sort(neigh.begin(), neigh.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
			return a.first < b.first;
		});

		for (const auto &kv : neigh) {
			adjncy.push_back((kahip_idx)kv.first);
			adjcwgt.push_back((kahip_idx)kv.second);
		}
	}
	xadj[(size_t)n] = (kahip_idx)adjncy.size();

	std::vector<int> vwgt((size_t)n, 1);
	for (int v = 0; v < n; ++v) {
		vwgt[(size_t)v] = clamp_positive_int(graph->vertex_compute_cost[v]);
	}

	int n_in = n;
	int nparts = k;
	double imbalance = config->hard_cap_ratio >= 0.0 ? config->hard_cap_ratio : 0.03;
	if (imbalance < 0.0) {
		imbalance = 0.03;
	}

	const bool suppress_output = true;
	const int seed = 42;
	const int mode = ECO;

	kahip_idx edgecut = 0;
	std::vector<int> part((size_t)n, 0);

	kaffpa(
		&n_in,
		vwgt.data(),
		xadj.data(),
		adjcwgt.data(),
		adjncy.data(),
		&nparts,
		&imbalance,
		suppress_output,
		seed,
		mode,
		&edgecut,
		part.data());

	for (int p = 0; p < k; ++p) {
		result->partition_loads[p] = 0.0;
	}
	for (int v = 0; v < n; ++v) {
		int pv = part[(size_t)v];
		if (pv < 0 || pv >= k) {
			pv = ((pv % k) + k) % k;
		}
		result->vertex_to_partition[v] = pv;
		result->partition_loads[pv] += graph->vertex_compute_cost[v];
	}

	double cut_cost = 0.0;
	for (int e = 0; e < graph->n_edges; ++e) {
		const int src = graph->edges[e].src;
		const int dst = graph->edges[e].dst;
		if (src < 0 || src >= n || dst < 0 || dst >= n) {
			continue;
		}
		if (result->vertex_to_partition[src] != result->vertex_to_partition[dst]) {
			cut_cost += graph->edges[e].weight;
		}
	}

	double total_load = 0.0;
	for (int v = 0; v < n; ++v) {
		total_load += graph->vertex_compute_cost[v];
	}
	const double target = total_load > 0.0 ? total_load / (double)k : 1.0;

	double balance_penalty = 0.0;
	for (int p = 0; p < k; ++p) {
		balance_penalty += load_penalty_single(result->partition_loads[p], target);
	}

	result->cut_cost = cut_cost;
	result->balance_penalty = balance_penalty;
	result->objective = config->lambda_comm * cut_cost + config->lambda_balance * balance_penalty;
	return true;
}
