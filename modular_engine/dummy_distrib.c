#include "distrib.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

static int find_graph_node_index(const dist_topology_node_t *nodes, int n_nodes, const struct ggml_tensor *target)
{
	if (nodes == NULL || target == NULL) {
		return -1;
	}

	for (int i = 0; i < n_nodes; ++i) {
		if (nodes[i].tensor == target) {
			return i;
		}
	}

	return -1;
}

static double estimate_tensor_cost(const struct ggml_tensor *t)
{
	if (t == NULL) {
		return 1.0;
	}

	double cost = (double) ggml_nbytes(t);
	const double n_elem = (double) ggml_nelements(t);

	switch (t->op) {
	case GGML_OP_MUL_MAT:
		cost *= 4.0 + 0.10 * log1p(n_elem);
		break;
	case GGML_OP_SOFT_MAX:
	case GGML_OP_DIAG_MASK_INF:
	case GGML_OP_RMS_NORM:
	case GGML_OP_NORM:
	case GGML_OP_UNARY:
		cost *= 1.5;
		break;
	default:
		break;
	}

	if (cost < 1.0) {
		cost = 1.0;
	}

	return cost;
}

bool dist_build_topology_tree_from_cgraph(const struct ggml_cgraph *graph, dist_topology_tree_t *tree)
{
	if (graph == NULL || tree == NULL) {
		return false;
	}

	memset(tree, 0, sizeof(*tree));

	const int n_nodes = ggml_graph_size((struct ggml_cgraph *) graph);
	if (n_nodes <= 0) {
		return false;
	}

	dist_topology_node_t *nodes = (dist_topology_node_t *) calloc((size_t) n_nodes, sizeof(*nodes));
	int *topo_order = (int *) malloc((size_t) n_nodes * sizeof(int));
	if (nodes == NULL || topo_order == NULL) {
		free(nodes);
		free(topo_order);
		return false;
	}

	for (int i = 0; i < n_nodes; ++i) {
		const struct ggml_tensor *tensor = ggml_graph_node((struct ggml_cgraph *) graph, i);
		nodes[i].tensor = tensor;
		nodes[i].tensor_index = i;
		nodes[i].parent = -1;
		nodes[i].first_child = -1;
		nodes[i].next_sibling = -1;
		nodes[i].depth = 0;
		nodes[i].compute_cost = estimate_tensor_cost(tensor);
		topo_order[i] = i;
	}

	int max_depth = 0;
	for (int i = 0; i < n_nodes; ++i) {
		const struct ggml_tensor *tensor = nodes[i].tensor;
		int best_parent = -1;
		int best_depth = 0;
		double best_cost = -1.0;

		for (int s = 0; s < GGML_MAX_SRC; ++s) {
			const struct ggml_tensor *src = tensor->src[s];
			if (src == NULL) {
				continue;
			}

			const int src_idx = find_graph_node_index(nodes, n_nodes, src);
			if (src_idx < 0) {
				continue;
			}

			const int candidate_depth = nodes[src_idx].depth + 1;
			const double candidate_cost = nodes[src_idx].compute_cost;
			if (candidate_depth > best_depth || (candidate_depth == best_depth && candidate_cost > best_cost)) {
				best_parent = src_idx;
				best_depth = candidate_depth;
				best_cost = candidate_cost;
			}
		}

		nodes[i].parent = best_parent;
		nodes[i].depth = best_parent >= 0 ? best_depth : 0;
		if (nodes[i].depth > max_depth) {
			max_depth = nodes[i].depth;
		}
	}

	for (int i = 0; i < n_nodes; ++i) {
		const int parent = nodes[i].parent;
		if (parent < 0) {
			continue;
		}

		nodes[i].next_sibling = nodes[parent].first_child;
		nodes[parent].first_child = i;
	}

	tree->n_nodes = n_nodes;
	tree->max_depth = max_depth;
	tree->nodes = nodes;
	tree->topo_order = topo_order;
	return true;
}

void dist_free_topology_tree(dist_topology_tree_t *tree)
{
	if (tree == NULL) {
		return;
	}

	free(tree->nodes);
	free(tree->topo_order);
	memset(tree, 0, sizeof(*tree));
}

void dist_print_topology_tree(const dist_topology_tree_t *tree, FILE *stream)
{
	if (tree == NULL || stream == NULL || tree->nodes == NULL) {
		return;
	}

	fprintf(stream, "[distrib] topology tree: nodes=%d max_depth=%d\n", tree->n_nodes, tree->max_depth);
	for (int i = 0; i < tree->n_nodes; ++i) {
		const dist_topology_node_t *node = &tree->nodes[i];
		const struct ggml_tensor *tensor = node->tensor;
		const char *op_name = tensor != NULL ? ggml_op_name(tensor->op) : "unknown";

		fprintf(stream,
				"[distrib] node[%d] depth=%d parent=%d op=%s cost=%.0f ne=[%lld,%lld,%lld,%lld]\n",
				i,
				node->depth,
				node->parent,
				op_name,
				node->compute_cost,
				tensor != NULL ? (long long) tensor->ne[0] : 0LL,
				tensor != NULL ? (long long) tensor->ne[1] : 0LL,
				tensor != NULL ? (long long) tensor->ne[2] : 0LL,
				tensor != NULL ? (long long) tensor->ne[3] : 0LL);
	}
}

bool dist_partition_topology_tree_levels(
	const dist_topology_tree_t *tree,
	int n_partitions,
	int *vertex_to_partition,
	double *partition_loads)
{
	if (tree == NULL || tree->nodes == NULL || vertex_to_partition == NULL || partition_loads == NULL) {
		return false;
	}
	if (n_partitions <= 0) {
		return false;
	}

	for (int p = 0; p < n_partitions; ++p) {
		partition_loads[p] = 0.0;
	}
	for (int i = 0; i < tree->n_nodes; ++i) {
		vertex_to_partition[i] = 0;
	}

	const int level_count = tree->max_depth + 1;
	double *level_costs = (double *) calloc((size_t) level_count, sizeof(double));
	if (level_costs == NULL) {
		return false;
	}

	double total_cost = 0.0;
	for (int i = 0; i < tree->n_nodes; ++i) {
		const int depth = tree->nodes[i].depth;
		level_costs[depth] += tree->nodes[i].compute_cost;
		total_cost += tree->nodes[i].compute_cost;
	}

	const double target = total_cost > 0.0 ? total_cost / (double) n_partitions : 1.0;
	int current_partition = 0;
	double current_load = 0.0;

	for (int depth = 0; depth < level_count; ++depth) {
		const double level_cost = level_costs[depth];
		const int remaining_levels = level_count - depth;
		const int remaining_parts = n_partitions - current_partition;

		if (current_partition < n_partitions - 1 && current_load > 0.0 && current_load + level_cost > target && remaining_levels > remaining_parts) {
			++current_partition;
			current_load = 0.0;
		}

		for (int i = 0; i < tree->n_nodes; ++i) {
			if (tree->nodes[i].depth != depth) {
				continue;
			}

			vertex_to_partition[i] = current_partition;
			partition_loads[current_partition] += tree->nodes[i].compute_cost;
		}

		current_load += level_cost;
	}

	free(level_costs);
	return true;
}

double dist_topology_tree_cut_cost(const dist_topology_tree_t *tree, const int *vertex_to_partition)
{
	if (tree == NULL || tree->nodes == NULL || vertex_to_partition == NULL) {
		return 0.0;
	}

	double cut_cost = 0.0;
	for (int i = 0; i < tree->n_nodes; ++i) {
		const int parent = tree->nodes[i].parent;
		if (parent < 0) {
			continue;
		}

		if (vertex_to_partition[parent] != vertex_to_partition[i]) {
			cut_cost += tree->nodes[i].compute_cost;
		}
	}

	return cut_cost;
}
