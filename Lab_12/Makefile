CC = gcc
CFLAGS = -O3 -Wall -fopenmp
LIBS = -lopenblas -lm

# Указываем пути явно, чтобы не путать с папкой OpenBLAS в корне
TARGET = blas_test

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LIBS)

test: $(TARGET)
	./$(TARGET)

test_fail: $(TARGET)
	./$(TARGET) --fail

clean:
	rm -f $(TARGET)
