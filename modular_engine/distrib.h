#ifndef MODULAR_ENGINE_DISTRIB_H
#define MODULAR_ENGINE_DISTRIB_H

#include <stdbool.h>
#include <stdio.h>

struct ggml_cgraph;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// Directed edge: src -> dst with communication cost weight.
typedef struct {
    int src;
    int dst;
    double weight;
} dist_edge_t;

typedef struct {
    int n_vertices;
    int n_edges;
    const double *vertex_compute_cost; // size: n_vertices
    const dist_edge_t *edges;          // size: n_edges
} dist_graph_t;

typedef struct {
    int n_partitions;

    // Objective = lambda_comm * cut_cost + lambda_balance * balance_penalty
    double lambda_comm;
    double lambda_balance;

    // Optional target load per partition (size n_partitions). If NULL, uses equal split.
    const double *target_partition_load;

    // Hard cap ratio over target. Example: 0.10 means load <= target * 1.10.
    // Set < 0 to disable hard capacity checks.
    double hard_cap_ratio;

    // Initialization/refinement tuning.
    int max_refine_iters;
    double min_delta_improvement; // stop when no move improves objective below -threshold
} dist_partition_config_t;

typedef struct {
    int *vertex_to_partition; // size n_vertices, caller-provided buffer
    double *partition_loads;  // size n_partitions, caller-provided buffer
    double cut_cost;
    double balance_penalty;
    double objective;
} dist_partition_result_t;

typedef struct dist_topology_node_t {
    const struct ggml_tensor *tensor;
    int tensor_index;
    int parent;
    int first_child;
    int next_sibling;
    int depth;
    double compute_cost;
} dist_topology_node_t;

typedef struct {
    int n_nodes;
    int max_depth;
    dist_topology_node_t *nodes;
    int *topo_order;
} dist_topology_tree_t;

// Returns true on success, false on invalid input.
// Buffers in result must be preallocated by caller.
bool dist_partition_graph_balanced_min_cut(
    const dist_graph_t *graph,
    const dist_partition_config_t *config,
    dist_partition_result_t *result);

// Build a lightweight topology tree from a ggml computation graph.
bool dist_build_topology_tree_from_cgraph(
    const struct ggml_cgraph *graph,
    dist_topology_tree_t *tree);

// Release memory owned by a topology tree.
void dist_free_topology_tree(dist_topology_tree_t *tree);

// Print a human-readable view of the tree.
void dist_print_topology_tree(const dist_topology_tree_t *tree, FILE *stream);

// Partition the tree by levels, keeping parent/child dependencies local when possible.
bool dist_partition_topology_tree_levels(
    const dist_topology_tree_t *tree,
    int n_partitions,
    int *vertex_to_partition,
    double *partition_loads);

// Approximate communication cost induced by cutting parent/child links across partitions.
double dist_topology_tree_cut_cost(
    const dist_topology_tree_t *tree,
    const int *vertex_to_partition);

// Export partition to DOT format for visualization.
// vertex_labels: optional array of strings (size n_vertices), can be NULL
// partition_names: optional array of partition names (size n_partitions), can be NULL
// Returns true on success.
bool dist_export_partition_to_dot(
    const char *output_path,
    const dist_graph_t *graph,
    const dist_partition_config_t *config,
    const dist_partition_result_t *result,
    const char *const *vertex_labels,
    const char *const *partition_names);

// Compute communication statistics per partition.
// Fills arrays (must be pre-allocated, size n_partitions each).
void dist_compute_partition_comm_stats(
    const dist_graph_t *graph,
    const int *vertex_to_partition,
    int n_partitions,
    double *outgoing_bytes,    // bytes leaving each partition
    double *incoming_bytes,    // bytes entering each partition
    double *internal_bytes);   // bytes within each partition

#ifdef __cplusplus
}
#endif

#endif
