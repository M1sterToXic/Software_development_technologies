all: blas_test

blas_test: main.c
	gcc -O3 -Wall -fopenmp main.c -o blas_test -lm

clean:
	rm -f blas_test
