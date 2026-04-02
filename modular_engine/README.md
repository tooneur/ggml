# try to modularisation of llama.cpp

this project is a try to modularize llama.cpp

this project is based on ggml. 

## Tensor visualizer

You can generate a Graphviz `.dot` view from the real content of `struct model`.

Default mode is generic (all loaded tensors + shapes).
Use `--gpt2` only when you want GPT-2 specific structural checks.

Build target:

```bash
cmake --build build --target visualizer
```

Run:

```bash
./build/bin/visualizer /path/to/model.gguf model-tensors.dot

# optional GPT-2 focused view
./build/bin/visualizer /path/to/model.gguf gpt2-tensors.dot --gpt2
```

Render to image:

```bash
dot -Tpng model-tensors.dot -o model-tensors.png
```

In `--gpt2` mode, the generated graph highlights:

- missing tensors (red)
- shape warnings (orange)
- expected tensors (blue)

This helps detect structural differences that can explain logits mismatches.

## Compare intermediate tensors from two models during inference

Build target:

```bash
cmake --build build --target model_diff
```

Run:

```bash
./build/bin/model_diff reference.gguf custom.gguf token_id [threshold]

# Examples:
./build/bin/model_diff reference.gguf custom.gguf 1          # threshold = 1e-4 (default)
./build/bin/model_diff reference.gguf custom.gguf 1 1e-6    # stricter threshold
```

**What it does:**
- Loads two models (reference and custom implementation)
- Runs inference on both with the same token_id
- Compares **final logits** element-wise
- Reports differences: max abs diff, mean diff, % of elements above threshold
- Returns exit code 0 if match, 1 if divergence detected

**Typical workflow:**
1. Build with `make model_diff` 
2. Run on a test token from reference GPT-2 (e.g., from OpenAI/Hugging Face)
3. Check output for where divergence starts
4. If logits differ significantly, the issue is likely in:
   - Tensor loading/shape mismatch
   - Attention computation
   - Feed-forward layer
   - Layer norm or bias handling

## Compare GGUF file structure (for completeness)

See `model_compare` tool above for comparing tensor metadata directly from files.


## Compare two GPT-2 GGUF models

Build target:

```bash
cmake --build build --target model_compare
```

Run:

```bash
./build/bin/model_compare /path/to/reference.gguf /path/to/custom.gguf

# optional: non-zero exit code if any structural/content mismatch is found
./build/bin/model_compare /path/to/reference.gguf /path/to/custom.gguf --strict
```

What is compared:

- GPT-2 hyperparameters (`n_layer`, `n_ctx`, `n_embd`, `n_ff`, `n_head`, `n_vocab`)
- tensor presence using canonical GPT-2 names (handles `transformer.h.*` and `blk.*` aliases)
- tensor type and shape
- tensor byte/content hash (all tensor types)
- element-wise value drift stats for `F32` tensors (`max_abs`, `mean_abs`)

we will try to follow thwe following configuration :

folder_name_of_the_model:
 - file for headers
 - file for metadata
 - file for index of all modules
 modules :
  - module_1.gguf
  - module_2.gguf
  - module_3.gguf


I think that the metadata will have the following format :

package_name: "my_modular_model_v1"
version: "1.0"
description: "Base + LoRA + head"
architecture: "transformer"
context_length: 2048
modules:
  - name: "base_model"
    type: "base"
    file: "modules/base_model.gguf"
  - name: "lora_1"
    type: "lora"
    target_layer: "layer_12"
    file: "modules/lora_1.gguf"
  - name: "head_addition"
    type: "head"
    target_layer: "output"
    file: "modules/head_addition.gguf"


# another try 

what we can do is to fully describe the architecture in a JSON/ custom format, such that we gain more in modularity

**the big question is how to store it such as the file isn't oversize ?** 

description of the custom format : 

package_name: "my_modular_model_v1"
version: "1.0"

then description of the architecture of the ML

finally description of the weight of the nodes in the ML.