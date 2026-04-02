#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ggml.h"
#include "gguf.h"
#include "parser.h"

struct report_state {
	int missing_tensors;
	int shape_warnings;
};

static struct ggml_tensor * find_tensor_any(
	struct ggml_context * ctx,
	const char * const * names,
	int n_names,
	const char ** matched_name) {

	for (int i = 0; i < n_names; ++i) {
		struct ggml_tensor * t = ggml_get_tensor(ctx, names[i]);
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

static void format_shape(const struct ggml_tensor * t, char * out, size_t out_size) {
	if (t == NULL) {
		snprintf(out, out_size, "missing");
		return;
	}

	snprintf(out, out_size, "[%lld x %lld x %lld x %lld] %s",
			 (long long) t->ne[0],
			 (long long) t->ne[1],
			 (long long) t->ne[2],
			 (long long) t->ne[3],
			 ggml_type_name(t->type));
}

static bool is_vector_n(const struct ggml_tensor * t, int64_t n) {
	return t != NULL &&
		   t->ne[0] == n &&
		   t->ne[1] == 1 &&
		   t->ne[2] == 1 &&
		   t->ne[3] == 1;
}

static bool is_matrix_pair(const struct ggml_tensor * t, int64_t a, int64_t b) {
	if (t == NULL) {
		return false;
	}
	if (t->ne[2] != 1 || t->ne[3] != 1) {
		return false;
	}
	return (t->ne[0] == a && t->ne[1] == b) ||
		   (t->ne[0] == b && t->ne[1] == a);
}

static bool is_matrix_exact(const struct ggml_tensor * t, int64_t a, int64_t b) {
	return t != NULL &&
		   t->ne[0] == a &&
		   t->ne[1] == b &&
		   t->ne[2] == 1 &&
		   t->ne[3] == 1;
}

static void emit_tensor_node(
	FILE * f,
	const char * node_id,
	const char * role,
	const char * resolved_name,
	const struct ggml_tensor * t,
	bool shape_ok,
	struct report_state * rep) {

	char shape_buf[128] = {0};
	format_shape(t, shape_buf, sizeof(shape_buf));

	const bool is_missing = (t == NULL);
	const char * border = "#24516b";
	const char * fill = "#dff3ff";

	if (is_missing) {
		border = "#8a1f1f";
		fill = "#ffdfe0";
		rep->missing_tensors += 1;
	} else if (!shape_ok) {
		border = "#8a5a00";
		fill = "#ffefcc";
		rep->shape_warnings += 1;
	}

	fprintf(f,
			"\"%s\" [shape=record, style=\"rounded,filled\", color=\"%s\", fillcolor=\"%s\", label=\"{%s|%s|%s}\"];\n",
			node_id,
			border,
			fill,
			role,
			resolved_name != NULL ? resolved_name : "MISSING",
			shape_buf);
}

static void emit_op_node(FILE * f, const char * node_id, const char * label) {
	fprintf(f,
			"\"%s\" [shape=box, style=\"rounded,filled\", color=\"#3b3b3b\", fillcolor=\"#f7f7f7\", label=\"%s\"];\n",
			node_id,
			label);
}

static void emit_edge(FILE * f, const char * src, const char * dst, const char * color, bool dashed) {
	fprintf(f,
			"\"%s\" -> \"%s\" [color=\"%s\"%s];\n",
			src,
			dst,
			color,
			dashed ? ", style=dashed" : "");
}

static bool load_block_tensor_any(
	struct ggml_context * ctx,
	int layer_idx,
	struct ggml_tensor ** out,
	const char ** matched,
	const char * blk_pattern,
	const char * alt_pattern,
	const char * alt_pattern_2) {

	char name_blk[128] = {0};
	char name_alt[128] = {0};
	char name_alt_2[128] = {0};
	const char * names[3] = {0};
	int n_names = 0;

	if (blk_pattern != NULL) {
		snprintf(name_blk, sizeof(name_blk), blk_pattern, layer_idx);
		names[n_names++] = name_blk;
	}

	if (alt_pattern != NULL) {
		snprintf(name_alt, sizeof(name_alt), alt_pattern, layer_idx);
		names[n_names++] = name_alt;
	}

	if (alt_pattern_2 != NULL) {
		snprintf(name_alt_2, sizeof(name_alt_2), alt_pattern_2, layer_idx);
		names[n_names++] = name_alt_2;
	}

	*out = find_tensor_any(ctx, names, n_names, matched);
	return *out != NULL;
}

static void write_dot_header(FILE * f, const struct model * model) {
	fprintf(f, "digraph GPT2_Tensors {\n");
	fprintf(f, "rankdir=LR;\n");
	fprintf(f, "labelloc=t;\n");
	fprintf(f, "label=\"GPT-2 Tensor Graph (%s)\";\n", model->model_name != NULL ? model->model_name : "unknown");
	fprintf(f, "fontname=\"Helvetica\";\n");
	fprintf(f, "node [fontname=\"Helvetica\", fontsize=10];\n");
	fprintf(f, "edge [fontname=\"Helvetica\", fontsize=9];\n");
}

static void dot_escape(const char * in, char * out, size_t out_size) {
	if (out_size == 0) {
		return;
	}

	size_t j = 0;
	for (size_t i = 0; in[i] != '\0' && j + 1 < out_size; ++i) {
		if (in[i] == '"' || in[i] == '\\') {
			if (j + 2 >= out_size) {
				break;
			}
			out[j++] = '\\';
		}
		out[j++] = in[i];
	}
	out[j] = '\0';
}

static void write_generic_dot(FILE * f, const struct model * model) {
	fprintf(f, "digraph Model_Tensors {\n");
	fprintf(f, "rankdir=TB;\n");
	fprintf(f, "labelloc=t;\n");
	fprintf(f, "label=\"Model Tensor Inventory (%s)\";\n", model->model_name != NULL ? model->model_name : "unknown");
	fprintf(f, "fontname=\"Helvetica\";\n");
	fprintf(f, "node [fontname=\"Helvetica\", fontsize=10];\n");

	fprintf(f, "subgraph cluster_meta {\n");
	fprintf(f, "label=\"Model Metadata\";\n");
	fprintf(f, "style=\"rounded\";\n");
	fprintf(f, "color=\"#9baec8\";\n");
	fprintf(f,
			"meta_hparams [shape=record, style=\"rounded,filled\", color=\"#4b5d76\", fillcolor=\"#eaf1fa\", "
			"label=\"{model|%s}|{n_tensors|%d}|{n_layer|%d}|{n_ctx|%d}|{n_embd|%d}|{n_ff|%d}|{n_head|%d}|{n_vocab|%d}\"];\n",
			model->model_name != NULL ? model->model_name : "unknown",
			model->n_tensors,
			model->hparams.n_layer,
			model->hparams.n_ctx,
			model->hparams.n_embd,
			model->hparams.n_ff,
			model->hparams.n_head,
			model->hparams.n_vocab);
	fprintf(f, "}\n");

	fprintf(f, "subgraph cluster_tensors {\n");
	fprintf(f, "label=\"Loaded Tensors\";\n");
	fprintf(f, "style=\"rounded\";\n");
	fprintf(f, "color=\"#d0d0d0\";\n");

	for (int i = 0; i < model->n_tensors; ++i) {
		const char * name = gguf_get_tensor_name(model->ctxgguf, i);
		const struct ggml_tensor * t = ggml_get_tensor(model->ctx, name);

		char escaped_name[256] = {0};
		dot_escape(name != NULL ? name : "<null>", escaped_name, sizeof(escaped_name));

		char shape_buf[128] = {0};
		if (t != NULL) {
			snprintf(shape_buf, sizeof(shape_buf), "[%lld x %lld x %lld x %lld] %s",
					 (long long) t->ne[0],
					 (long long) t->ne[1],
					 (long long) t->ne[2],
					 (long long) t->ne[3],
					 ggml_type_name(t->type));
		} else {
			snprintf(shape_buf, sizeof(shape_buf), "missing in ggml context");
		}

		fprintf(f,
				"t_%d [shape=record, style=\"rounded,filled\", color=\"#2f4f4f\", fillcolor=\"#eef9f1\", label=\"{%d|%s|%s}\"];\n",
				i,
				i,
				escaped_name,
				shape_buf);
	}

	fprintf(f, "}\n");
	fprintf(f, "}\n");
}

int main(int argc, char ** argv) {
	if (argc < 2) {
		printf("Usage: %s model.gguf [output.dot] [--gpt2]\n", argv[0]);
		return 1;
	}

	const char * model_path = argv[1];
	bool gpt2_mode = false;

	const char * out_path = "model-tensors.dot";
	int option_start = 2;
	if (argc >= 3 && strncmp(argv[2], "--", 2) != 0) {
		out_path = argv[2];
		option_start = 3;
	}

	for (int i = option_start; i < argc; ++i) {
		if (strcmp(argv[i], "--gpt2") == 0) {
			gpt2_mode = true;
			if (strcmp(out_path, "model-tensors.dot") == 0) {
				out_path = "gpt2-tensors.dot";
			}
		} else {
			printf("Unknown option: %s\n", argv[i]);
			printf("Usage: %s model.gguf [output.dot] [--gpt2]\n", argv[0]);
			return 1;
		}
	}

	struct model model = {0};
	if (!model_load(model_path, &model)) {
		printf("Failed to load model: %s\n", model_path);
		return 1;
	}

	FILE * f = fopen(out_path, "w");
	if (f == NULL) {
		printf("Failed to open output file: %s\n", out_path);
		gguf_free(model.ctxgguf);
		return 1;
	}

	if (!gpt2_mode) {
		write_generic_dot(f, &model);
		fclose(f);

		printf("Generic DOT graph generated: %s\n", out_path);
		printf("Model: %s\n", model.model_name != NULL ? model.model_name : "unknown");
		printf("n_tensors: %d\n", model.n_tensors);
		printf("hparams: n_layer=%d n_ctx=%d n_embd=%d n_ff=%d n_head=%d n_vocab=%d\n",
			   model.hparams.n_layer,
			   model.hparams.n_ctx,
			   model.hparams.n_embd,
			   model.hparams.n_ff,
			   model.hparams.n_head,
			   model.hparams.n_vocab);
		printf("Tip: use --gpt2 to enable GPT-2 structural checks.\n");
		printf("To render: dot -Tpng %s -o model-tensors.png\n", out_path);

		gguf_free(model.ctxgguf);
		return 0;
	}

	struct report_state rep = {0};
	const int64_t n_embd = model.hparams.n_embd;
	const int64_t n_vocab = model.hparams.n_vocab;
	const int64_t n_ctx = model.hparams.n_ctx;
	const int64_t n_ff = model.hparams.n_ff;

	write_dot_header(f, &model);

	fprintf(f, "subgraph cluster_global {\n");
	fprintf(f, "label=\"Global Tensors\";\n");
	fprintf(f, "color=\"#b0cbe0\";\n");
	fprintf(f, "style=\"rounded\";\n");

	const char * m_wte = NULL;
	const char * n_wte[] = {"token_embd.weight", "transformer.wte.weight"};
	struct ggml_tensor * t_wte = find_tensor_any(model.ctx, n_wte, 2, &m_wte);
	emit_tensor_node(f, "wte", "wte", m_wte, t_wte, is_matrix_pair(t_wte, n_embd, n_vocab), &rep);

	const char * m_wpe = NULL;
	const char * n_wpe[] = {"position_embd.weight", "transformer.wpe.weight"};
	struct ggml_tensor * t_wpe = find_tensor_any(model.ctx, n_wpe, 2, &m_wpe);
	emit_tensor_node(f, "wpe", "wpe", m_wpe, t_wpe, is_matrix_exact(t_wpe, n_embd, n_ctx), &rep);

	const char * m_lnf_g = NULL;
	const char * n_lnf_g[] = {"output_norm.weight", "transformer.ln_f.weight"};
	struct ggml_tensor * t_lnf_g = find_tensor_any(model.ctx, n_lnf_g, 2, &m_lnf_g);
	emit_tensor_node(f, "ln_f_g", "ln_f.weight", m_lnf_g, t_lnf_g, is_vector_n(t_lnf_g, n_embd), &rep);

	const char * m_lnf_b = NULL;
	const char * n_lnf_b[] = {"output_norm.bias", "transformer.ln_f.bias"};
	struct ggml_tensor * t_lnf_b = find_tensor_any(model.ctx, n_lnf_b, 2, &m_lnf_b);
	emit_tensor_node(f, "ln_f_b", "ln_f.bias", m_lnf_b, t_lnf_b, is_vector_n(t_lnf_b, n_embd), &rep);

	const char * m_lm = NULL;
	const char * n_lm[] = {"output.weight", "lm_head.weight"};
	struct ggml_tensor * t_lm = find_tensor_any(model.ctx, n_lm, 2, &m_lm);
	emit_tensor_node(f, "lm_head", "lm_head", m_lm, t_lm, is_matrix_pair(t_lm, n_embd, n_vocab), &rep);

	fprintf(f, "}\n");

	emit_op_node(f, "embed", "token + position embd");
	emit_edge(f, "wte", "embed", "#2c7a4b", false);
	emit_edge(f, "wpe", "embed", "#2c7a4b", false);

	char prev_x[32] = {0};
	snprintf(prev_x, sizeof(prev_x), "x0");
	emit_op_node(f, prev_x, "x0");
	emit_edge(f, "embed", prev_x, "#3b3b3b", false);

	for (int i = 0; i < model.hparams.n_layer; ++i) {
		fprintf(f, "subgraph cluster_layer_%d {\n", i);
		fprintf(f, "label=\"Block %d\";\n", i);
		fprintf(f, "color=\"#d0d0d0\";\n");
		fprintf(f, "style=\"rounded\";\n");

		struct ggml_tensor * t_ln1_g = NULL;
		struct ggml_tensor * t_ln1_b = NULL;
		struct ggml_tensor * t_attn_w = NULL;
		struct ggml_tensor * t_attn_b = NULL;
		struct ggml_tensor * t_proj_w = NULL;
		struct ggml_tensor * t_proj_b = NULL;
		struct ggml_tensor * t_ln2_g = NULL;
		struct ggml_tensor * t_ln2_b = NULL;
		struct ggml_tensor * t_fc_w = NULL;
		struct ggml_tensor * t_fc_b = NULL;
		struct ggml_tensor * t_proj2_w = NULL;
		struct ggml_tensor * t_proj2_b = NULL;

		const char * m_ln1_g = NULL;
		const char * m_ln1_b = NULL;
		const char * m_attn_w = NULL;
		const char * m_attn_b = NULL;
		const char * m_proj_w = NULL;
		const char * m_proj_b = NULL;
		const char * m_ln2_g = NULL;
		const char * m_ln2_b = NULL;
		const char * m_fc_w = NULL;
		const char * m_fc_b = NULL;
		const char * m_proj2_w = NULL;
		const char * m_proj2_b = NULL;

		load_block_tensor_any(model.ctx, i, &t_ln1_g, &m_ln1_g,
							  "blk.%d.attn_norm.weight",
							  "transformer.h.%d.ln_1.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_ln1_b, &m_ln1_b,
							  "blk.%d.attn_norm.bias",
							  "transformer.h.%d.ln_1.bias",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_attn_w, &m_attn_w,
							  "blk.%d.attn_qkv.weight",
							  "transformer.h.%d.attn.c_attn.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_attn_b, &m_attn_b,
							  "blk.%d.attn_qkv.bias",
							  "transformer.h.%d.attn.c_attn.bias",
							  "transformer.h.%d.attn.c_attn.b");
		load_block_tensor_any(model.ctx, i, &t_proj_w, &m_proj_w,
							  "blk.%d.attn_output.weight",
							  "transformer.h.%d.attn.c_proj.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_proj_b, &m_proj_b,
							  "blk.%d.attn_output.bias",
							  "transformer.h.%d.attn.c_proj.bias",
							  "transformer.h.%d.attn.c_proj.b");
		load_block_tensor_any(model.ctx, i, &t_ln2_g, &m_ln2_g,
							  "blk.%d.ffn_norm.weight",
							  "transformer.h.%d.ln_2.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_ln2_b, &m_ln2_b,
							  "blk.%d.ffn_norm.bias",
							  "transformer.h.%d.ln_2.bias",
							  "transformer.h.%d.ln_2.b");
		load_block_tensor_any(model.ctx, i, &t_fc_w, &m_fc_w,
							  "blk.%d.ffn_up.weight",
							  "transformer.h.%d.mlp.c_fc.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_fc_b, &m_fc_b,
							  "blk.%d.ffn_up.bias",
							  "transformer.h.%d.mlp.c_fc.bias",
							  "transformer.h.%d.mlp.c_fc.b");
		load_block_tensor_any(model.ctx, i, &t_proj2_w, &m_proj2_w,
							  "blk.%d.ffn_down.weight",
							  "transformer.h.%d.mlp.c_proj.weight",
							  NULL);
		load_block_tensor_any(model.ctx, i, &t_proj2_b, &m_proj2_b,
							  "blk.%d.ffn_down.bias",
							  "transformer.h.%d.mlp.c_proj.bias",
							  "transformer.h.%d.mlp.c_proj.b");

		char id_ln1_g[32], id_ln1_b[32], id_attn_w[32], id_attn_b[32], id_proj_w[32], id_proj_b[32];
		char id_ln2_g[32], id_ln2_b[32], id_fc_w[32], id_fc_b[32], id_proj2_w[32], id_proj2_b[32];
		char id_ln1[32], id_attn[32], id_res1[32], id_ln2[32], id_fc[32], id_gelu[32], id_proj2[32], id_res2[32], id_xnext[32];

		snprintf(id_ln1_g, sizeof(id_ln1_g), "L%d_ln1_g", i);
		snprintf(id_ln1_b, sizeof(id_ln1_b), "L%d_ln1_b", i);
		snprintf(id_attn_w, sizeof(id_attn_w), "L%d_attn_w", i);
		snprintf(id_attn_b, sizeof(id_attn_b), "L%d_attn_b", i);
		snprintf(id_proj_w, sizeof(id_proj_w), "L%d_proj_w", i);
		snprintf(id_proj_b, sizeof(id_proj_b), "L%d_proj_b", i);
		snprintf(id_ln2_g, sizeof(id_ln2_g), "L%d_ln2_g", i);
		snprintf(id_ln2_b, sizeof(id_ln2_b), "L%d_ln2_b", i);
		snprintf(id_fc_w, sizeof(id_fc_w), "L%d_fc_w", i);
		snprintf(id_fc_b, sizeof(id_fc_b), "L%d_fc_b", i);
		snprintf(id_proj2_w, sizeof(id_proj2_w), "L%d_proj2_w", i);
		snprintf(id_proj2_b, sizeof(id_proj2_b), "L%d_proj2_b", i);

		snprintf(id_ln1, sizeof(id_ln1), "L%d_ln1", i);
		snprintf(id_attn, sizeof(id_attn), "L%d_attn", i);
		snprintf(id_res1, sizeof(id_res1), "L%d_res1", i);
		snprintf(id_ln2, sizeof(id_ln2), "L%d_ln2", i);
		snprintf(id_fc, sizeof(id_fc), "L%d_fc", i);
		snprintf(id_gelu, sizeof(id_gelu), "L%d_gelu", i);
		snprintf(id_proj2, sizeof(id_proj2), "L%d_proj2", i);
		snprintf(id_res2, sizeof(id_res2), "L%d_res2", i);
		snprintf(id_xnext, sizeof(id_xnext), "x%d", i + 1);

		emit_tensor_node(f, id_ln1_g, "ln_1.weight", m_ln1_g, t_ln1_g, is_vector_n(t_ln1_g, n_embd), &rep);
		emit_tensor_node(f, id_ln1_b, "ln_1.bias", m_ln1_b, t_ln1_b, is_vector_n(t_ln1_b, n_embd), &rep);
		emit_tensor_node(f, id_attn_w, "c_attn.weight", m_attn_w, t_attn_w, is_matrix_pair(t_attn_w, n_embd, 3 * n_embd), &rep);
		emit_tensor_node(f, id_attn_b, "c_attn.bias", m_attn_b, t_attn_b, is_vector_n(t_attn_b, 3 * n_embd), &rep);
		emit_tensor_node(f, id_proj_w, "attn.c_proj.weight", m_proj_w, t_proj_w, is_matrix_pair(t_proj_w, n_embd, n_embd), &rep);
		emit_tensor_node(f, id_proj_b, "attn.c_proj.bias", m_proj_b, t_proj_b, is_vector_n(t_proj_b, n_embd), &rep);
		emit_tensor_node(f, id_ln2_g, "ln_2.weight", m_ln2_g, t_ln2_g, is_vector_n(t_ln2_g, n_embd), &rep);
		emit_tensor_node(f, id_ln2_b, "ln_2.bias", m_ln2_b, t_ln2_b, is_vector_n(t_ln2_b, n_embd), &rep);
		emit_tensor_node(f, id_fc_w, "mlp.c_fc.weight", m_fc_w, t_fc_w, is_matrix_pair(t_fc_w, n_embd, n_ff), &rep);
		emit_tensor_node(f, id_fc_b, "mlp.c_fc.bias", m_fc_b, t_fc_b, is_vector_n(t_fc_b, n_ff), &rep);
		emit_tensor_node(f, id_proj2_w, "mlp.c_proj.weight", m_proj2_w, t_proj2_w, is_matrix_pair(t_proj2_w, n_ff, n_embd), &rep);
		emit_tensor_node(f, id_proj2_b, "mlp.c_proj.bias", m_proj2_b, t_proj2_b, is_vector_n(t_proj2_b, n_embd), &rep);

		emit_op_node(f, id_ln1, "LayerNorm 1");
		emit_op_node(f, id_attn, "Self-Attn (QKV + softmax)");
		emit_op_node(f, id_res1, "Residual Add");
		emit_op_node(f, id_ln2, "LayerNorm 2");
		emit_op_node(f, id_fc, "MLP c_fc");
		emit_op_node(f, id_gelu, "GELU");
		emit_op_node(f, id_proj2, "MLP c_proj");
		emit_op_node(f, id_res2, "Residual Add");
		emit_op_node(f, id_xnext, id_xnext);

		emit_edge(f, id_ln1_g, id_ln1, "#1f5c8a", true);
		emit_edge(f, id_ln1_b, id_ln1, "#1f5c8a", true);
		emit_edge(f, id_attn_w, id_attn, "#1f5c8a", true);
		emit_edge(f, id_attn_b, id_attn, "#1f5c8a", true);
		emit_edge(f, id_proj_w, id_attn, "#1f5c8a", true);
		emit_edge(f, id_proj_b, id_attn, "#1f5c8a", true);
		emit_edge(f, id_ln2_g, id_ln2, "#1f5c8a", true);
		emit_edge(f, id_ln2_b, id_ln2, "#1f5c8a", true);
		emit_edge(f, id_fc_w, id_fc, "#1f5c8a", true);
		emit_edge(f, id_fc_b, id_fc, "#1f5c8a", true);
		emit_edge(f, id_proj2_w, id_proj2, "#1f5c8a", true);
		emit_edge(f, id_proj2_b, id_proj2, "#1f5c8a", true);

		emit_edge(f, prev_x, id_ln1, "#3b3b3b", false);
		emit_edge(f, id_ln1, id_attn, "#3b3b3b", false);
		emit_edge(f, id_attn, id_res1, "#3b3b3b", false);
		emit_edge(f, prev_x, id_res1, "#6f4aa1", false);
		emit_edge(f, id_res1, id_ln2, "#3b3b3b", false);
		emit_edge(f, id_ln2, id_fc, "#3b3b3b", false);
		emit_edge(f, id_fc, id_gelu, "#3b3b3b", false);
		emit_edge(f, id_gelu, id_proj2, "#3b3b3b", false);
		emit_edge(f, id_proj2, id_res2, "#3b3b3b", false);
		emit_edge(f, id_res1, id_res2, "#6f4aa1", false);
		emit_edge(f, id_res2, id_xnext, "#3b3b3b", false);

		fprintf(f, "}\n");

		snprintf(prev_x, sizeof(prev_x), "%s", id_xnext);
	}

	emit_op_node(f, "ln_f", "Final LayerNorm");
	emit_op_node(f, "logits", "Logits");

	emit_edge(f, "ln_f_g", "ln_f", "#1f5c8a", true);
	emit_edge(f, "ln_f_b", "ln_f", "#1f5c8a", true);
	emit_edge(f, prev_x, "ln_f", "#3b3b3b", false);
	emit_edge(f, "ln_f", "logits", "#3b3b3b", false);
	emit_edge(f, "lm_head", "logits", "#1f5c8a", true);

	fprintf(f, "}\n");
	fclose(f);

	printf("DOT graph generated: %s\n", out_path);
	printf("Hyperparams: n_layer=%d n_ctx=%d n_embd=%d n_head=%d n_ff=%d n_vocab=%d\n",
		   model.hparams.n_layer,
		   model.hparams.n_ctx,
		   model.hparams.n_embd,
		   model.hparams.n_head,
		   model.hparams.n_ff,
		   model.hparams.n_vocab);
	printf("Missing tensors: %d\n", rep.missing_tensors);
	printf("Shape warnings: %d\n", rep.shape_warnings);

	if (rep.missing_tensors > 0 || rep.shape_warnings > 0) {
		printf("Model deviates from base GPT-2 tensor layout. Inspect highlighted nodes in DOT output.\n");
	} else {
		printf("Tensor layout matches expected GPT-2 structure (name aliases + shape checks).\n");
	}

	printf("To render: dot -Tpng %s -o gpt2-tensors.png\n", out_path);

	gguf_free(model.ctxgguf);
	return 0;
}
