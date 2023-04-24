#include <stdlib.h>
#include <stdio.h>

// code=transpose && nvcc -o $code.o $code.cu && ./$code.o
// code=transpose && nvcc -arch=sm_35 -o $code.o $code.cu && ./$code.o

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void d_transpose(int* d_mat1, int* d_mat2, int dim1, int dim2)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (yIndex < dim1 && xIndex < dim2)
   {
       unsigned int index_in  = yIndex * dim2 + xIndex;
       unsigned int index_out = xIndex * dim1 + yIndex;
       d_mat2[index_out] = d_mat1[index_in]; 
   }
}

void h_transpose(int* mat1, int* mat2, int dim1, int dim2) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            mat2[x * dim1 + y] = mat1[y * dim2 + x];
        }
    }
}

void transpose(int* mat1, int* mat2, int dim1, int dim2) {
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    if (*deviceCount == 0) {
        h_transpose(mat1, mat2, dim1, dim2);
    } else {
        int size = dim1 * dim2;
        int BLOCK_DIM = 16;
        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 blocksPerGrid(
            (dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (dim2 + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        printf(
            "threadsPerBlock.x=%d, threadsPerBlock.y=%d, blocksPerGrid.x=%d, blocksPerGrid.y=%d\n",
            threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y
        );

        int* d_mat1;
        int* d_mat2;

        gpuErrchk(cudaMalloc((void**) &d_mat1, size * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat2, size * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_mat1, mat1, size * sizeof(int), cudaMemcpyHostToDevice));

        d_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, dim1, dim2);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(mat2, d_mat2, size * sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_mat1));
        gpuErrchk(cudaFree(d_mat2));
    }
    printf("deviceCount = %d\n", *deviceCount);
    free(deviceCount);
}

void initialize(int* mat, int dim1, int dim2) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            mat[y * dim2 + x] = y * dim2 + x;
        }
    }
}

void printRow(int* row, int size) {
    printf("[");
    for (int x = 0; x < size - 1; x++) {
        printf("%d, ", row[x]);
    }
    if (size != 0) printf("%d", row[size - 1]);
    printf("]\n");
}

void print(int* mat, int dim1, int dim2) {
    printf("[\n");
    for (int y = 0; y < dim1; y++) {
        printf("\t");
        printRow(&mat[y * dim2], dim2);
    }
    printf("]\n");
}

int main(int argc, char** argv) {
    int dim1 = 8;
    int dim2 = 8;
    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim2 = (int) atoi(argv[2]);

    int SIZE = dim1 * dim2;

    int* mat1 = (int*) malloc(SIZE * sizeof(int));
    int* mat2 = (int*) malloc(SIZE * sizeof(int));

    initialize(mat1, dim1, dim2);

    transpose(mat1, mat2, dim1, dim2);

    print(mat1, dim1, dim2);
    print(mat2, dim2, dim1);

    free(mat1);
    free(mat2);

    return 0;
}