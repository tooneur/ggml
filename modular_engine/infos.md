gguf format : 

[GGUF]
[version]
[nb-tensor]
[nb entree metadata]

# type C typique
struct gguf_header {
    char     magic[4];   
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

[archi]
[context]
[tokenizer]


tensor infos :
- tensor name
- tensor shape

struct gguf_tensor_info {
    string name;
    uint32_t n_dims;
    uint64_t dims[n_dims];
    ggml_type type;
    uint64_t offset;
}


struct gguf_tensor_info {
    string name;
    uint32_t n_dims;
    uint64_t dims[n_dims];
    ggml_type type;
    uint64_t offset;
}

tensors data:

contient les poids

