OPENBLAS_DIR = /mnt/c/Users/krivo/Tecnology_PO/openblas

CC = gcc
CFLAGS = -Wall -g
INCLUDES = -I$(OPENBLAS_DIR)
LIBS = -L$(OPENBLAS_DIR) -lopenblas -lm

TARGET = level2_tests
SOURCE = level2_tests.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	LD_LIBRARY_PATH=$(OPENBLAS_DIR) ./$(TARGET)