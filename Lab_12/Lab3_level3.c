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

void dsymm_seq(char side, char uplo, int m, int n,
               double alpha, const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
    int i, j, k;
    double temp;
    if (m == 0 || n == 0) return;
    if (fabs(beta - 1.0) > 1e-14) {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = beta * C[i + j*ldc];
    } else if (fabs(beta) < 1e-14) {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = 0.0;
    }
    if (fabs(alpha) < 1e-14) return;
    if (side == 'L' || side == 'l') {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0;
                for (k = 0; k < m; ++k) {
                    double a;
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
                temp = 0.0;
                for (k = 0; k < n; ++k) {
                    double a;
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

void dsymm_omp(char side, char uplo, int m, int n,
               double alpha, const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
    int i, j, k;
    double temp;
    if (m == 0 || n == 0) return;
    if (fabs(beta - 1.0) > 1e-14) {
        #pragma omp parallel for collapse(2)
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = beta * C[i + j*ldc];
    } else if (fabs(beta) < 1e-14) {
        #pragma omp parallel for collapse(2)
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                C[i + j*ldc] = 0.0;
    }
    if (fabs(alpha) < 1e-14) return;
    if (side == 'L' || side == 'l') {
        #pragma omp parallel for private(i, k, temp)
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                temp = 0.0;
                for (k = 0; k < m; ++k) {
                    double a;
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
                temp = 0.0;
                for (k = 0; k < n; ++k) {
                    double a;
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

void random_matrix_double(int m, int n, double *mat, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mat[i + j*lda] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

int validate_float(int m, int n, float *C, float *C_ref, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (fabs(C[i + j*ldc] - C_ref[i + j*ldc]) > 1e-20) {
                printf("Validation failed at (%d,%d): My=%.6f, Ref=%.6f\n", i, j, C[i + j*ldc], C_ref[i + j*ldc]);
                return 0;
            }
        }
    }
    return 1;
}

int validate_double(int m, int n, double *C, double *C_ref, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (fabs(C[i + j*ldc] - C_ref[i + j*ldc]) > 1e-10) {
                printf("Validation failed at (%d,%d): My=%.12f, Ref=%.12f\n", i, j, C[i + j*ldc], C_ref[i + j*ldc]);
                return 0;
            }
        }
    }
    return 1;
}

int test_ssymm(int m, int n, int num_threads, int parallel, int performance_only) {
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

    cblas_ssymm(CblasColMajor,
                (side=='L'?CblasLeft:CblasRight),
                (uplo=='U'?CblasUpper:CblasLower),
                m, n, alpha, A, lda, B, ldb, beta, C_ref, ldc);

    if (!validate_float(m, n, C, C_ref, ldc)) {
        free(A); free(B); free(C); free(C_ref);
        return 0;
    }

    if (performance_only) {
        double time_my_sum = 0.0, time_ref_sum = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            double t1 = get_time();
            if (parallel) ssymm_omp(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
            else ssymm_seq(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
            time_my_sum += (get_time() - t1);

            t1 = get_time();
            cblas_ssymm(CblasColMajor, (side=='L'?CblasLeft:CblasRight), (uplo=='U'?CblasUpper:CblasLower),
                        m, n, alpha, A, lda, B, ldb, beta, C_ref, ldc);
            time_ref_sum += (get_time() - t1);
        }
        printf("ssymm %s %d threads: my=%.3fs, ref=%.3fs, ratio=%.2f%%\n",
               parallel?"omp":"seq", num_threads, time_my_sum, time_ref_sum, (time_ref_sum/time_my_sum)*100.0);
    }

    free(A); free(B); free(C); free(C_ref);
    return 1;
}

int test_dsymm(int m, int n, int num_threads, int parallel, int performance_only) {
    omp_set_num_threads(num_threads);

    char side = 'L';
    char uplo = 'U';
    double alpha = 1.0;
    double beta = 0.0;

    int lda = (side == 'L' ? m : n);
    int ldb = m;
    int ldc = m;

    double *A, *B, *C, *C_ref;
    A = (double*)aligned_alloc(64, lda * lda * sizeof(double));
    B = (double*)aligned_alloc(64, ldb * n * sizeof(double));
    C = (double*)aligned_alloc(64, ldc * n * sizeof(double));
    C_ref = (double*)aligned_alloc(64, ldc * n * sizeof(double));

    srand(12345);
    random_matrix_double(lda, lda, A, lda);
    random_matrix_double(ldb, n, B, ldb);
    random_matrix_double(ldc, n, C, ldc);
    memcpy(C_ref, C, ldc * n * sizeof(double));

    if (parallel)
        dsymm_omp(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        dsymm_seq(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

    cblas_dsymm(CblasColMajor,
                (side=='L'?CblasLeft:CblasRight),
                (uplo=='U'?CblasUpper:CblasLower),
                m, n, alpha, A, lda, B, ldb, beta, C_ref, ldc);

    if (!validate_double(m, n, C, C_ref, ldc)) {
        free(A); free(B); free(C); free(C_ref);
        return 0;
    }

    if (performance_only) {
        double time_my_sum = 0.0, time_ref_sum = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            double t1 = get_time();
            if (parallel) dsymm_omp(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
            else dsymm_seq(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
            time_my_sum += (get_time() - t1);

            t1 = get_time();
            cblas_dsymm(CblasColMajor, (side=='L'?CblasLeft:CblasRight), (uplo=='U'?CblasUpper:CblasLower),
                        m, n, alpha, A, lda, B, ldb, beta, C_ref, ldc);
            time_ref_sum += (get_time() - t1);
        }
        printf("dsymm %s %d threads: my=%.3fs, ref=%.3fs, ratio=%.2f%%\n",
               parallel?"omp":"seq", num_threads, time_my_sum, time_ref_sum, (time_ref_sum/time_my_sum)*100.0);
    }

    free(A); free(B); free(C); free(C_ref);
    return 1;
}

int main(int argc, char **argv) {
    int performance_only = 0;
    if (argc > 1 && strcmp(argv[1], "--performance") == 0) {
        performance_only = 1;
    }

    int m = 1000;
    int n = 1000;
    int threads[] = {1, 2, 4};
    int nthreads = sizeof(threads)/sizeof(threads[0]);

    if (!performance_only) printf("Running interface tests (correctness)...\n");

    for (int i = 0; i < nthreads; ++i) {
        if (!test_ssymm(m, n, threads[i], 0, performance_only)) return 1;
        if (!test_ssymm(m, n, threads[i], 1, performance_only)) return 1;
        if (!test_dsymm(m, n, threads[i], 0, performance_only)) return 1;
        if (!test_dsymm(m, n, threads[i], 1, performance_only)) return 1;
    }

    if (!performance_only) printf("All interface tests passed!\n");
    return 0;
}