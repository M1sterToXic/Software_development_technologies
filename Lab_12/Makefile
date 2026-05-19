CC = gcc
CFLAGS = -O3 -Wall -fopenmp
LIBS = -lopenblas -lm

all: blas_test

blas_test: main.c
	$(CC) $(CFLAGS) main.c -o blas_test $(LIBS)

test: blas_test
	./blas_test

test_fail: blas_test
	./blas_test --fail

clean:
	rm -f blas_test
