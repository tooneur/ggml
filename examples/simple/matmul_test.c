#include "ggml.h"
#include "ggml-cpu.h"
#include "string.h"
#include "ggml-backend.h"


int main() {

    // initialisation de la mémoire ATTENTION ggml demande à ce que toute la mémoire soit défini en avance car c'est lui qui gère les mallocs
    struct ggml_init_params params =
    {
        .mem_size = 16*1024*1024,
        .mem_buffer = NULL,

    };

    // initialize data of matrices to perform matrix multiplication
    #define ROWS_A 4
    #define COLS_A 2

    float matrix_A[ROWS_A * COLS_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    #define ROWS_B 2
    #define COLS_B 3

    float matrix_B[ROWS_B * COLS_B] = {
        10, 5, 9,
        9,  5, 4
    };

    // appel de l'init
    struct ggml_context * ctx = ggml_init(params);

    // def des tensors
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);

    //copy des données
    memcpy(A->data, matrix_A, sizeof(matrix_A));
    memcpy(B->data, matrix_B, sizeof(matrix_B));

    struct ggml_tensor * C = ggml_mul_mat(ctx, B, A);

    // construction du graphe de calcul
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    //execution
    // Créer backend CPU
    ggml_backend_t backend = ggml_backend_cpu_init();
    

    
    ggml_backend_graph_compute(backend, gf);
    
    ggml_backend_free(backend);


    float * c = (float *)C->data;
    for (int i = 0; i < 8; i++)
        printf("%f\n", c[i]);

    //libération de la mémoire OBLIGATOIRE vu que c'est ggml qui gere les free
    ggml_free(ctx);

    return 0;

}

