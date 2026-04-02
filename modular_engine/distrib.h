#ifndef MODULAR_ENGINE_DISTRIB_H
#define MODULAR_ENGINE_DISTRIB_H

#include <stdbool.h>

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

// Returns true on success, false on invalid input.
// Buffers in result must be preallocated by caller.
bool dist_partition_graph_balanced_min_cut(
    const dist_graph_t *graph,
    const dist_partition_config_t *config,
    dist_partition_result_t *result);

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
