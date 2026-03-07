#define _GNU_SOURCE
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
    if (fabs(beta - 1.0f) > 1e-7) {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = beta * C[i + j*ldc];
    } else if (fabs(beta) < 1e-7) {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = 0.0f;
    }
    if (fabs(alpha) < 1e-7) return;
    if (side == 'L' || side == 'l') {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0f;
                for (k = 0; k < m; ++k) {
                    float a;
                    if (uplo == 'U' || uplo == 'u') {
                        if (i <= k) a = A[i + k*lda];
                        else        a = A[k + i*lda];
                    } else {
                        if (i >= k) a = A[i + k*lda];
                        else        a = A[k + i*lda];
                    }
                    temp += a * B[k + j*ldb];
                }
                C[i + j*ldc] += alpha * temp;
            }
        }
    } else {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0f;
                for (k = 0; k < n; ++k) {
                    float a;
                    if (uplo == 'U' || uplo == 'u') {
                        if (j <= k) a = A[j + k*lda];
                        else        a = A[k + j*lda];
                    } else {
                        if (j >= k) a = A[j + k*lda];
                        else        a = A[k + j*lda];
                    }
                    temp += B[i + k*ldb] * a;
                }
                C[i + j*ldc] += alpha * temp;
            }
        }
    }
}

void ssymm_omp(char side, char uplo, int m, int n,
               float alpha, const float *A, int lda,
               const float *B, int ldb,
               float beta, float *C, int ldc) {
    int i, j, k;
    float temp;
    if (m == 0 || n == 0) return;
    if (fabs(beta - 1.0f) > 1e-7) {
        #pragma omp parallel for collapse(2)
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = beta * C[i + j*ldc];
    } else if (fabs(beta) < 1e-7) {
        #pragma omp parallel for collapse(2)
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = 0.0f;
    }
    if (fabs(alpha) < 1e-7) return;
    if (side == 'L' || side == 'l') {
        #pragma omp parallel for private(i, k, temp)
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0f;
                for (k = 0; k < m; ++k) {
                    float a;
                    if (uplo == 'U' || uplo == 'u') {
                        if (i <= k) a = A[i + k*lda];
                        else        a = A[k + i*lda];
                    } else {
                        if (i >= k) a = A[i + k*lda];
                        else        a = A[k + i*lda];
                    }
                    temp += a * B[k + j*ldb];
                }
                C[i + j*ldc] += alpha * temp;
            }
        }
    } else {
        #pragma omp parallel for private(i, k, temp)
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0f;
                for (k = 0; k < n; ++k) {
                    float a;
                    if (uplo == 'U' || uplo == 'u') {
                        if (j <= k) a = A[j + k*lda];
                        else        a = A[k + j*lda];
                    } else {
                        if (j >= k) a = A[j + k*lda];
                        else        a = A[k + j*lda];
                    }
                    temp += B[i + k*ldb] * a;
                }
                C[i + j*ldc] += alpha * temp;
            }
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void random_matrix_float(int m, int n, float *mat, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mat[i + j*lda] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

void test_ssymm(int m, int n, int num_threads, int parallel) {
    omp_set_num_threads(num_threads);

    char side = 'L';
    char uplo = 'U';
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = (side == 'L' ? m : n);
    int ldb = m;
    int ldc = m;

    float *A, *B, *C, *C_ref;
    A = (float*)aligned_alloc(64, lda * lda * sizeof(float));
    B = (float*)aligned_alloc(64, ldb * n * sizeof(float));
    C = (float*)aligned_alloc(64, ldc * n * sizeof(float));
    C_ref = (float*)aligned_alloc(64, ldc * n * sizeof(float));

    srand(12345);
    random_matrix_float(lda, lda, A, lda);
    random_matrix_float(ldb, n, B, ldb);
    random_matrix_float(ldc, n, C, ldc);
    memcpy(C_ref, C, ldc * n * sizeof(float));

    if (parallel)
        ssymm_omp(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        ssymm_seq(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

    double perf_values[10];
    double time_my_sum = 0.0, time_ref_sum = 0.0;

    for (int rep = 0; rep < 10; ++rep) {
        memset(C, 0, ldc * n * sizeof(float));
        double t1 = get_time();
        if (parallel)
            ssymm_omp(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            ssymm_seq(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        double t2 = get_time();
        double time_my = t2 - t1;
        time_my_sum += time_my;

        memset(C_ref, 0, ldc * n * sizeof(float));
        t1 = get_time();
        cblas_ssymm(CblasColMajor,
                    (side=='L'?CblasLeft:CblasRight),
                    (uplo=='U'?CblasUpper:CblasLower),
                    m, n, alpha, A, lda, B, ldb, beta, C_ref, ldc);
        t2 = get_time();
        double time_ref = t2 - t1;
        time_ref_sum += time_ref;

        perf_values[rep] = (time_ref / time_my) * 100.0;
    }

    double log_sum = 0.0;
    for (int rep = 0; rep < 10; ++rep)
        log_sum += log(perf_values[rep]);
    double geom_mean = exp(log_sum / 10.0);

    printf("%2d потоков: моё время (сумма 10 запусков) = %.3f с, OpenBLAS (сумма) = %.3f с, средняя относит. = %.2f%%, геом. среднее = %.2f%%\n",
           num_threads, time_my_sum, time_ref_sum, (time_ref_sum/time_my_sum)*100.0, geom_mean);

    free(A); free(B); free(C); free(C_ref);
}

int main() {
    int m = 2000;
    int n = 2000;
    int threads[] = {1, 2, 4, 8, 16};
    int nthreads = sizeof(threads)/sizeof(threads[0]);

    printf("===========================================================\n");
    printf("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ SYMM (BLAS уровень 3)\n");
    printf("Размер матриц: %d x %d\n", m, n);
    printf("===========================================================\n\n");

    printf("------------ float (ssymm) - последовательная версия ------------\n");
    for (int i = 0; i < nthreads; ++i) {
        test_ssymm(m, n, threads[i], 0);
    }
    printf("\n------------ float (ssymm) - параллельная версия (OpenMP) ------------\n");
    for (int i = 0; i < nthreads; ++i) {
        test_ssymm(m, n, threads[i], 1);
    }

    return 0;
}