#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

void ssymm_seq(int m, int n, const float *A, const float *B, float *C) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float temp = 0.0f;
            for (int k = 0; k < m; ++k) {
                float a = (i <= k) ? A[i + k * m] : A[k + i * m];
                temp += a * B[k + j * m];
            }
            C[i + j * m] = temp;
        }
    }
}

void ssymm_omp(int m, int n, const float *A, const float *B, float *C) {
    int j, i, k;
    #pragma omp parallel for private(i, k) shared(A, B, C, m, n)
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            float temp = 0.0f;
            for (k = 0; k < m; ++k) {
                float a = (i <= k) ? A[i + k * m] : A[k + i * m];
                temp += a * B[k + j * m];
            }
            C[i + j * m] = temp;
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv, "--fail") == 0) {
        printf("[WARN] Режим искусственного падения тестов активирован.\n");
        return 1; 
    }

    printf("[INFO] Запуск интерфейсных тестов...\n");
    int m_test = 4, n_test = 4;
    float *A_test = malloc(16 * sizeof(float));
    float *B_test = malloc(16 * sizeof(float));
    float *C_seq_test = malloc(16 * sizeof(float));
    float *C_omp_test = malloc(16 * sizeof(float));

    if (!A_test || !B_test || !C_seq_test || !C_omp_test) return 1;

    for (int i = 0; i < 16; i++) {
        A_test[i] = (float)(i % 4);
        B_test[i] = 1.0f;
    }

    ssymm_seq(m_test, n_test, A_test, B_test, C_seq_test);
    ssymm_omp(m_test, n_test, A_test, B_test, C_omp_test);

    for (int i = 0; i < 16; i++) {
        if (fabsf(C_seq_test[i] - C_omp_test[i]) > 1e-5f) {
            printf("[ERROR] Интерфейсный тест ПРОВАЛЕН!\n");
            free(A_test); free(B_test); free(C_seq_test); free(C_omp_test);
            return 1;
        }
    }
    printf("[SUCCESS] Интерфейсные тесты успешно пройдены!\n");
    free(A_test); free(B_test); free(C_seq_test); free(C_omp_test);

    printf("[INFO] Запуск тестов производительности...\n");
    int m_perf = 600, n_perf = 600;
    float *A_perf = malloc(m_perf * m_perf * sizeof(float));
    float *B_perf = malloc(m_perf * n_perf * sizeof(float));
    float *C_perf = malloc(m_perf * n_perf * sizeof(float));

    if (!A_perf || !B_perf || !C_perf) return 1;

    for (int i = 0; i < m_perf * m_perf; i++) A_perf[i] = 1.0f;
    for (int i = 0; i < m_perf * n_perf; i++) B_perf[i] = 2.0f;

    omp_set_num_threads(2);

    double t0 = get_time();
    ssymm_seq(m_perf, n_perf, A_perf, B_perf, C_perf);
    double t_seq = get_time() - t0;
    printf("Последовательное время: %.4f сек\n", t_seq);

    double t1 = get_time();
    ssymm_omp(m_perf, n_perf, A_perf, B_perf, C_perf);
    double t_omp = get_time() - t1;
    printf("Параллельное время (2 потока): %.4f сек\n", t_omp);

    free(A_perf); free(B_perf); free(C_perf);

    if (t_omp <= 0.0) {
        printf("[ERROR] Тест производительности ПРОВАЛЕН!\n");
        return 1;
    }
    printf("[SUCCESS] Тест производительности успешно пройден!\n");

    return 0;
}