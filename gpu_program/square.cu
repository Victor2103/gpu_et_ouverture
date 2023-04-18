#include <stdio.h>
#include <stdlib.h>

// code=square && nvcc -o $code.o $code.cu && ./$code.o
// code=square && nvcc -arch=sm_35 -o $code.o $code.cu && ./$code.o

#define PRINTIT true

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __global__ void d_square_loop(int* d_a, int* d_b, int size) {
//     for (int i = 0; i < size; i++) {
//         d_b[i] = d_a[i] * d_a[i];
//     }
// }

__global__ void d_square_oneeach(int* d_a, int* d_b, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        d_b[index] = d_a[index] * d_a[index];
    }
}

void h_square(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        b[i] = a[i] * a[i];
    }
}

void square(int* a, int* b, int size) {
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    if (*deviceCount == 0) {
        h_square(a, b, size);
    } else {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        printf(
            "threadsPerBlock=%d, blocksPerGrid=%d\n",
            threadsPerBlock, blocksPerGrid
        );

        int* d_a;
        int* d_b;

        gpuErrchk(cudaMalloc((void**) &d_a, size * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_b, size * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice));

        // d_square_loop<<<1, 1>>>(d_a, d_b, size);
        // gpuErrchk(cudaPeekAtLastError());

        d_square_oneeach<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, size);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(b, d_b, size * sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_a));
        gpuErrchk(cudaFree(d_b));
    }
    printf("deviceCount = %d\n", *deviceCount);
    free(deviceCount);
}

void initialize(int* v, int size) {
    for (int i = 0; i < size; i++) {
        v[i] = i;
    }
}

void print(int* v, int size) {
    printf("[");
    for (int i = 0; i < size - 1; i++) {
        printf("%d, ", v[i]);
    }
    if (size != 0) printf("%d", v[size - 1]);
    printf("]\n");
}

int main(int argc, char** argv) {
    int SIZE = 32;
    if (argc > 1) SIZE = (int) atoi(argv[1]);
    printf("SIZE=%d\n", SIZE);

    int* a = (int*) malloc(SIZE * sizeof(int));
    int* b = (int*) malloc(SIZE * sizeof(int));

    initialize(a, SIZE);
    printf("a = ");
    print(a, SIZE);

    square(a, b, SIZE);

    printf("b = ");
    print(b, SIZE);

    free(a);
    free(b);

    return 0;
}