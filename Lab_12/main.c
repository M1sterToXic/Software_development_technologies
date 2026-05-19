#include <stdio.h>
#include <stdlib.h>
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

int run_interface_tests() {
    printf("[INFO] Запуск интерфейсных тестов...\n");
    int m = 4, n = 4;
    float A, B, C_seq, C_omp;
    
    for (int i = 0; i < 16; i++) {
        A[i] = (float)(i % 4);
        B[i] = 1.0f;
    }

    ssymm_seq(m, n, A, B, C_seq);
    ssymm_omp(m, n, A, B, C_omp);

    for (int i = 0; i < 16; i++) {
        if (fabsf(C_seq[i] - C_omp[i]) > 1e-5f) {
            printf("[ERROR] Интерфейсный тест ПРОВАЛЕН! Расхождение в элементе %d\n", i);
            return 1; 
        }
    }
    printf("[SUCCESS] Интерфейсные тесты успешно пройдены!\n");
    return 0;
}

int run_performance_tests() {
    printf("[INFO] Запуск тестов производительности...\n");
    int m = 600, n = 600;
    float *A = malloc(m * m * sizeof(float));
    float *B = malloc(m * n * sizeof(float));
    float *C = malloc(m * n * sizeof(float));

    if (!A || !B || !C) return 1;

    for (int i = 0; i < m*m; i++) A[i] = 1.0f;
    for (int i = 0; i < m*n; i++) B[i] = 2.0f;

    omp_set_num_threads(2);

    double t0 = get_time();
    ssymm_seq(m, n, A, B, C);
    double t_seq = get_time() - t0;
    printf("Последовательное время: %.4f сек\n", t_seq);

    double t1 = get_time();
    ssymm_omp(m, n, A, B, C);
    double t_omp = get_time() - t1;
    printf("Параллельное время (2 потока): %.4f сек\n", t_omp);

    free(A); free(B); free(C);

    if (t_omp <= 0.0) {
        printf("[ERROR] Тест производительности ПРОВАЛЕН! Некорректное время.\n");
        return 1;
    }
    printf("[SUCCESS] Тест производительности успешно пройден!\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv, "--fail") == 0) {
        printf("[WARN] Режим искусственного падения тестов активирован.\n");
        return 1; 
    }

    int res1 = run_interface_tests();
    int res2 = run_performance_tests();

    return (res1 == 0 && res2 == 0) ? 0 : 1;
}