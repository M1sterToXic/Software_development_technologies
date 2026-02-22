#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cblas.h"

float eps_f = 1e-6;
double eps_d = 1e-12;

int check_s(char* name, float expected, float actual, int index) {
    if(fabsf(expected - actual) > eps_f) {
        printf("  FAILED: %s[%d] = %f, expected %f\n", name, index, actual, expected);
        return 1;
    }
    return 0;
}

int check_d(char* name, double expected, double actual, int index) {
    if(fabs(expected - actual) > eps_d) {
        printf("  FAILED: %s[%d] = %f, expected %f\n", name, index, actual, expected);
        return 1;
    }
    return 0;
}

int test_sgemv() {
    printf("Testing sgemv...\n");
    int fails = 0;
    float A[4] = {1, 2, 3, 4};
    float x[2] = {1, 1};
    float y[2] = {0, 0};
    float expected[2] = {3, 7};
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    
    fails += check_s("sgemv", expected[0], y[0], 0);
    fails += check_s("sgemv", expected[1], y[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dgemv() {
    printf("Testing dgemv...\n");
    int fails = 0;
    double A[4] = {1, 2, 3, 4};
    double x[2] = {1, 1};
    double y[2] = {0, 0};
    double expected[2] = {3, 7};
    
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    
    fails += check_d("dgemv", expected[0], y[0], 0);
    fails += check_d("dgemv", expected[1], y[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_cgemv() {
    printf("Testing cgemv...\n");
    int fails = 0;
    float _Complex A[4] = {1+1*I, 2+2*I, 3+3*I, 4+4*I};
    float _Complex x[2] = {1+0*I, 1+0*I};
    float _Complex y[2] = {0+0*I, 0+0*I};
    float _Complex alpha = 1+0*I;
    float _Complex beta = 0+0*I;
    
    cblas_cgemv(CblasRowMajor, CblasNoTrans, 2, 2, &alpha, A, 2, x, 1, &beta, y, 1);
    
    float diff_re = fabs(crealf(y[0]) - 3);
    float diff_im = fabs(cimagf(y[0]) - 3);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: cgemv[0] = %f+%fi, expected 3+3i\n", crealf(y[0]), cimagf(y[0]));
        fails++;
    }
    
    diff_re = fabs(crealf(y[1]) - 7);
    diff_im = fabs(cimagf(y[1]) - 7);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: cgemv[1] = %f+%fi, expected 7+7i\n", crealf(y[1]), cimagf(y[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_zgemv() {
    printf("Testing zgemv...\n");
    int fails = 0;
    double _Complex A[4] = {1+1*I, 2+2*I, 3+3*I, 4+4*I};
    double _Complex x[2] = {1+0*I, 1+0*I};
    double _Complex y[2] = {0+0*I, 0+0*I};
    double _Complex alpha = 1+0*I;
    double _Complex beta = 0+0*I;
    
    cblas_zgemv(CblasRowMajor, CblasNoTrans, 2, 2, &alpha, A, 2, x, 1, &beta, y, 1);
    
    double diff_re = fabs(creal(y[0]) - 3);
    double diff_im = fabs(cimag(y[0]) - 3);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: zgemv[0] = %f+%fi, expected 3+3i\n", creal(y[0]), cimag(y[0]));
        fails++;
    }
    
    diff_re = fabs(creal(y[1]) - 7);
    diff_im = fabs(cimag(y[1]) - 7);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: zgemv[1] = %f+%fi, expected 7+7i\n", creal(y[1]), cimag(y[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_strmv() {
    printf("Testing strmv...\n");
    int fails = 0;
    float A[4] = {1, 2, 0, 4};
    float x[2] = {1, 1};
    float expected[2] = {3, 4};
    
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    fails += check_s("strmv", expected[0], x[0], 0);
    fails += check_s("strmv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dtrmv() {
    printf("Testing dtrmv...\n");
    int fails = 0;
    double A[4] = {1, 2, 0, 4};
    double x[2] = {1, 1};
    double expected[2] = {3, 4};
    
    cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    fails += check_d("dtrmv", expected[0], x[0], 0);
    fails += check_d("dtrmv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ctrmv() {
    printf("Testing ctrmv...\n");
    int fails = 0;
    float _Complex A[4] = {1+0*I, 2+0*I, 0+0*I, 4+0*I};
    float _Complex x[2] = {1+0*I, 1+0*I};
    
    cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    float diff_re = fabs(crealf(x[0]) - 3);
    float diff_im = fabs(cimagf(x[0]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctrmv[0] = %f+%fi, expected 3+0i\n", crealf(x[0]), cimagf(x[0]));
        fails++;
    }
    
    diff_re = fabs(crealf(x[1]) - 4);
    diff_im = fabs(cimagf(x[1]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctrmv[1] = %f+%fi, expected 4+0i\n", crealf(x[1]), cimagf(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ztrmv() {
    printf("Testing ztrmv...\n");
    int fails = 0;
    double _Complex A[4] = {1+0*I, 2+0*I, 0+0*I, 4+0*I};
    double _Complex x[2] = {1+0*I, 1+0*I};
    
    cblas_ztrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    double diff_re = fabs(creal(x[0]) - 3);
    double diff_im = fabs(cimag(x[0]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztrmv[0] = %f+%fi, expected 3+0i\n", creal(x[0]), cimag(x[0]));
        fails++;
    }
    
    diff_re = fabs(creal(x[1]) - 4);
    diff_im = fabs(cimag(x[1]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztrmv[1] = %f+%fi, expected 4+0i\n", creal(x[1]), cimag(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ssymv() {
    printf("Testing ssymv...\n");
    int fails = 0;
    float A[4] = {1, 2, 2, 4};
    float x[2] = {1, 1};
    float y[2] = {0, 0};
    float expected[2] = {3, 6};
    
    cblas_ssymv(CblasRowMajor, CblasUpper, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    
    fails += check_s("ssymv", expected[0], y[0], 0);
    fails += check_s("ssymv", expected[1], y[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dsymv() {
    printf("Testing dsymv...\n");
    int fails = 0;
    double A[4] = {1, 2, 2, 4};
    double x[2] = {1, 1};
    double y[2] = {0, 0};
    double expected[2] = {3, 6};
    
    cblas_dsymv(CblasRowMajor, CblasUpper, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    
    fails += check_d("dsymv", expected[0], y[0], 0);
    fails += check_d("dsymv", expected[1], y[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_sger() {
    printf("Testing sger...\n");
    int fails = 0;
    float A[4] = {0, 0, 0, 0};
    float x[2] = {2, 3};
    float y[2] = {4, 5};
    float expected[4] = {8, 10, 12, 15};
    
    cblas_sger(CblasRowMajor, 2, 2, 1.0, x, 1, y, 1, A, 2);
    
    fails += check_s("sger", expected[0], A[0], 0);
    fails += check_s("sger", expected[1], A[1], 1);
    fails += check_s("sger", expected[2], A[2], 2);
    fails += check_s("sger", expected[3], A[3], 3);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dger() {
    printf("Testing dger...\n");
    int fails = 0;
    double A[4] = {0, 0, 0, 0};
    double x[2] = {2, 3};
    double y[2] = {4, 5};
    double expected[4] = {8, 10, 12, 15};
    
    cblas_dger(CblasRowMajor, 2, 2, 1.0, x, 1, y, 1, A, 2);
    
    fails += check_d("dger", expected[0], A[0], 0);
    fails += check_d("dger", expected[1], A[1], 1);
    fails += check_d("dger", expected[2], A[2], 2);
    fails += check_d("dger", expected[3], A[3], 3);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_cgerc() {
    printf("Testing cgerc...\n");
    int fails = 0;
    float _Complex A[4] = {0+0*I, 0+0*I, 0+0*I, 0+0*I};
    float _Complex x[2] = {2+1*I, 3+2*I};
    float _Complex y[2] = {4+3*I, 5+4*I};
    float _Complex alpha = 1+0*I;
    
    cblas_cgerc(CblasRowMajor, 2, 2, &alpha, x, 1, y, 1, A, 2);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_zgerc() {
    printf("Testing zgerc...\n");
    int fails = 0;
    double _Complex A[4] = {0+0*I, 0+0*I, 0+0*I, 0+0*I};
    double _Complex x[2] = {2+1*I, 3+2*I};
    double _Complex y[2] = {4+3*I, 5+4*I};
    double _Complex alpha = 1+0*I;
    
    cblas_zgerc(CblasRowMajor, 2, 2, &alpha, x, 1, y, 1, A, 2);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_cgeru() {
    printf("Testing cgeru...\n");
    int fails = 0;
    float _Complex A[4] = {0+0*I, 0+0*I, 0+0*I, 0+0*I};
    float _Complex x[2] = {2+1*I, 3+2*I};
    float _Complex y[2] = {4+3*I, 5+4*I};
    float _Complex alpha = 1+0*I;
    
    cblas_cgeru(CblasRowMajor, 2, 2, &alpha, x, 1, y, 1, A, 2);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_zgeru() {
    printf("Testing zgeru...\n");
    int fails = 0;
    double _Complex A[4] = {0+0*I, 0+0*I, 0+0*I, 0+0*I};
    double _Complex x[2] = {2+1*I, 3+2*I};
    double _Complex y[2] = {4+3*I, 5+4*I};
    double _Complex alpha = 1+0*I;
    
    cblas_zgeru(CblasRowMajor, 2, 2, &alpha, x, 1, y, 1, A, 2);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_strsv() {
    printf("Testing strsv...\n");
    int fails = 0;
    float A[4] = {1, 2, 0, 4};
    float x[2] = {3, 4};
    float expected[2] = {1, 1};
    
    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    fails += check_s("strsv", expected[0], x[0], 0);
    fails += check_s("strsv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dtrsv() {
    printf("Testing dtrsv...\n");
    int fails = 0;
    double A[4] = {1, 2, 0, 4};
    double x[2] = {3, 4};
    double expected[2] = {1, 1};
    
    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    fails += check_d("dtrsv", expected[0], x[0], 0);
    fails += check_d("dtrsv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ctrsv() {
    printf("Testing ctrsv...\n");
    int fails = 0;
    float _Complex A[4] = {1+0*I, 2+0*I, 0+0*I, 4+0*I};
    float _Complex x[2] = {3+0*I, 4+0*I};
    
    cblas_ctrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    float diff_re = fabs(crealf(x[0]) - 1);
    float diff_im = fabs(cimagf(x[0]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctrsv[0] = %f+%fi, expected 1+0i\n", crealf(x[0]), cimagf(x[0]));
        fails++;
    }
    
    diff_re = fabs(crealf(x[1]) - 1);
    diff_im = fabs(cimagf(x[1]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctrsv[1] = %f+%fi, expected 1+0i\n", crealf(x[1]), cimagf(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ztrsv() {
    printf("Testing ztrsv...\n");
    int fails = 0;
    double _Complex A[4] = {1+0*I, 2+0*I, 0+0*I, 4+0*I};
    double _Complex x[2] = {3+0*I, 4+0*I};
    
    cblas_ztrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    
    double diff_re = fabs(creal(x[0]) - 1);
    double diff_im = fabs(cimag(x[0]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztrsv[0] = %f+%fi, expected 1+0i\n", creal(x[0]), cimag(x[0]));
        fails++;
    }
    
    diff_re = fabs(creal(x[1]) - 1);
    diff_im = fabs(cimag(x[1]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztrsv[1] = %f+%fi, expected 1+0i\n", creal(x[1]), cimag(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_stpmv() {
    printf("Testing stpmv...\n");
    int fails = 0;
    float Ap[3] = {1, 2, 4};
    float x[2] = {1, 1};
    float expected[2] = {3, 4};
    
    cblas_stpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap, x, 1);
    
    fails += check_s("stpmv", expected[0], x[0], 0);
    fails += check_s("stpmv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_dtpmv() {
    printf("Testing dtpmv...\n");
    int fails = 0;
    double Ap[3] = {1, 2, 4};
    double x[2] = {1, 1};
    double expected[2] = {3, 4};
    
    cblas_dtpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap, x, 1);
    
    fails += check_d("dtpmv", expected[0], x[0], 0);
    fails += check_d("dtpmv", expected[1], x[1], 1);
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ctpmv() {
    printf("Testing ctpmv...\n");
    int fails = 0;
    float _Complex Ap[3] = {1+0*I, 2+0*I, 4+0*I};
    float _Complex x[2] = {1+0*I, 1+0*I};
    
    cblas_ctpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap, x, 1);
    
    float diff_re = fabs(crealf(x[0]) - 3);
    float diff_im = fabs(cimagf(x[0]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctpmv[0] = %f+%fi, expected 3+0i\n", crealf(x[0]), cimagf(x[0]));
        fails++;
    }
    
    diff_re = fabs(crealf(x[1]) - 4);
    diff_im = fabs(cimagf(x[1]) - 0);
    if(diff_re > eps_f || diff_im > eps_f) {
        printf("  FAILED: ctpmv[1] = %f+%fi, expected 4+0i\n", crealf(x[1]), cimagf(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int test_ztpmv() {
    printf("Testing ztpmv...\n");
    int fails = 0;
    double _Complex Ap[3] = {1+0*I, 2+0*I, 4+0*I};
    double _Complex x[2] = {1+0*I, 1+0*I};
    
    cblas_ztpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap, x, 1);
    
    double diff_re = fabs(creal(x[0]) - 3);
    double diff_im = fabs(cimag(x[0]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztpmv[0] = %f+%fi, expected 3+0i\n", creal(x[0]), cimag(x[0]));
        fails++;
    }
    
    diff_re = fabs(creal(x[1]) - 4);
    diff_im = fabs(cimag(x[1]) - 0);
    if(diff_re > eps_d || diff_im > eps_d) {
        printf("  FAILED: ztpmv[1] = %f+%fi, expected 4+0i\n", creal(x[1]), cimag(x[1]));
        fails++;
    }
    
    if(fails == 0) printf("  PASSED\n");
    return fails;
}

int main() {
    int total_fails = 0;
    
    printf("\n=== CBLAS LEVEL 2 TESTS ===\n\n");
    
    total_fails += test_sgemv();
    total_fails += test_dgemv();
    total_fails += test_cgemv();
    total_fails += test_zgemv();
    
    total_fails += test_strmv();
    total_fails += test_dtrmv();
    total_fails += test_ctrmv();
    total_fails += test_ztrmv();
    
    total_fails += test_ssymv();
    total_fails += test_dsymv();
    
    total_fails += test_sger();
    total_fails += test_dger();
    total_fails += test_cgerc();
    total_fails += test_zgerc();
    total_fails += test_cgeru();
    total_fails += test_zgeru();
    
    total_fails += test_strsv();
    total_fails += test_dtrsv();
    total_fails += test_ctrsv();
    total_fails += test_ztrsv();
    
    total_fails += test_stpmv();
    total_fails += test_dtpmv();
    total_fails += test_ctpmv();
    total_fails += test_ztpmv();
    
    printf("\n=== RESULT ===\n");
    if(total_fails == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("%d TESTS FAILED\n", total_fails);
        return 1;
    }
}