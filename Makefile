
PROJECT_NAME = test_add
CFLAGS = -c -fPIC
NVCCHOST = gcc-7
NVCCFLAGS = -ccbin $(NVCCHOST) -c
LFLAGS =
CUDAPATH = /usr/local/cuda


NVCC = nvcc
CC = clang

all: build clean

run: build
	./a.out

build: gpu
	$(NVCC) $(LFLAGS) *.o

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu:
	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

profile:
	nvprof ./a.out
