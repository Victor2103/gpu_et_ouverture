#include <stdio.h>
#include <stdlib.h>

// code=somme && nvcc -o $code.o $code.cu && ./$code.o
// code=somme && nvcc -arch=sm_35 -o $code.o $code.cu && ./$code.o

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

__global__ void d_sum_oneeach(int* d_a, int* d_b, int* d_c,int size) {
    int index = blockIdx.x * blockDim.x*2 + threadIdx.x;
    if (index < size) {
        d_c[index] = d_a[index] + d_b[index];
    }
}

void h_sum(int* a, int* b, int* c,int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void sum(int* a, int* b,int* c, int size) {
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    if (*deviceCount == 0) {
        h_sum(a, b,c, size);
    } else {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        printf(
            "threadsPerBlock=%d, blocksPerGrid=%d\n",
            threadsPerBlock, blocksPerGrid
        );

        int* d_a;
        int* d_b;
        int* d_c;

        gpuErrchk(cudaMalloc((void**) &d_a, size * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_b, size * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_c, size * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
        // d_square_loop<<<1, 1>>>(d_a, d_b, size);
        // gpuErrchk(cudaPeekAtLastError());

        d_sum_oneeach<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c,size);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_a));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_c));
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
    int* c = (int*) malloc(SIZE * sizeof(int));

    initialize(a, SIZE);
    printf("a = ");
    print(a, SIZE);

    initialize(b, SIZE);
    printf("b = ");
    print(b, SIZE);

    sum(a, b,c, SIZE);

    printf("c = ");
    print(c, SIZE);

    free(a);
    free(b);

    return 0;
}