CC = gcc
CFLAGS = -O3 -Wall -fopenmp
LIBS = -lopenblas -lm

TARGET = blas_test
SRC = Lab3_level3.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

test: $(TARGET)
	./$(TARGET) --interface

perf: $(TARGET)
	./$(TARGET) --performance

clean:
	rm -f $(TARGET)
