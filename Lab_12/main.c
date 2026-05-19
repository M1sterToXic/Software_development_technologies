#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>

void ssymm_seq(char side, char uplo, int m, int n,
               float alpha, const float *A, int lda,
               const float *B, int ldb,
               float beta, float *C, int ldc) {
    int i, j, k;
    float temp;
    if (m == 0 || n == 0) return;
    for (j = 0; j < n; ++j)
        for (i = 0; i < m; ++i)
            C[i + j*ldc] = beta * C[i + j*ldc];

    if (side == 'L' || side == 'l') {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0f;
                for (k = 0; k < m; ++k) {
                    float a = (uplo == 'U' || uplo == 'u') ? 
                              (i <= k ? A[i + k*lda] : A[k + i*lda]) :
                              (i >= k ? A[i + k*lda] : A[k + i*lda]);
                    temp += a * B[k + j*ldb];
                }
                C[i + j*ldc] += alpha * temp;
            }
        }
    }
}

int main(int argc, char **argv) {
    int m = 500, n = 500;
    float *A = malloc(m * m * sizeof(float));
    float *B = malloc(m * n * sizeof(float));
    float *C = malloc(m * n * sizeof(float));
    float *C_ref = malloc(m * n * sizeof(float));

    for(int i=0; i<m*m; i++) A[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<m*n; i++) B[i] = (float)rand()/RAND_MAX;
    memset(C, 0, m * n * sizeof(float));
    memset(C_ref, 0, m * n * sizeof(float));

    printf("Running BLAS Interface Test...\n");
    ssymm_seq('L', 'U', m, n, 1.0f, A, m, B, m, 0.0f, C, m);
    cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, m, n, 1.0f, A, m, B, m, 0.0f, C_ref, m);

    float max_diff = 0.0f;
    for(int i=0; i<m*n; i++) {
        float diff = fabsf(C[i] - C_ref[i]);
        if(diff > max_diff) max_diff = diff;
    }

    printf("Max difference: %f\n", max_diff);
    
    float threshold = 1e-4f;
    if (argc > 1 && strcmp(argv[1], "--fail") == 0) threshold = 1e-20f;

    if(max_diff > threshold) {
        printf("TEST FAILED!\n");
        return 1;
    }

    printf("TEST PASSED!\n");
    return 0;
}
