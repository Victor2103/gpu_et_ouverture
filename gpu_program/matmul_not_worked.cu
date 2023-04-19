#include <stdlib.h>
#include <stdio.h>

// code=multiplication && nvcc -o $code.o $code.cu && ./$code.o
// code=multiplication && nvcc -arch=sm_35 -o $code.o $code.cu && ./$code.o

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void d_matmul(int* d_mat1, int* d_mat2, int* d_out, int dim1, int dim2, int dim_s)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   unsigned int kIndex = blockDim.z * blockIdx.z + threadIdx.z;
   if (yIndex < dim1 && xIndex < dim2)
   {
        int sum=d_out[yIndex * dim2 + xIndex];
        if (kIndex < dim_s) {
           sum+= d_mat1[yIndex * dim_s + kIndex] * d_mat2[kIndex * dim2 + xIndex];
       }
       d_out[yIndex * dim2 + xIndex]=sum;
    }
}

void h_matmul(int* mat1, int* mat2, int* out, int dim1, int dim2, int dim_s) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            int o = 0;
            for (int k = 0; k < dim_s; k++) {
                o += mat1[y * dim_s + k] * mat2[k * dim2 + x];
            }
            out[y * dim2 + x] = o;
        }
    }
}

void matmul(int* mat1, int* mat2, int* mat3, int dim1, int dim2,int dim_s) {
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    if (*deviceCount == 0) {
        h_matmul(mat1, mat2, mat3, dim1, dim2, dim_s);
    } else {
        int size = dim1 * dim2;
        int BLOCK_DIM = 32;
        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 blocksPerGrid(
            (dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (dim2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (dim_s + threadsPerBlock.z - 1) / threadsPerBlock.z
        );
        printf(
            "threadsPerBlock.x=%d, threadsPerBlock.y=%d, blocksPerGrid.x=%d, blocksPerGrid.y=%d\n",
            threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y
        );

        int* d_mat1;
        int* d_mat2;
        int* d_mat3;

        gpuErrchk(cudaMalloc((void**) &d_mat1, dim1 * dim_s * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat2, dim_s * dim2 * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat3, size * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_mat1, mat1, dim1 * dim_s * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mat2, mat2, dim_s * dim2 * sizeof(int), cudaMemcpyHostToDevice));

        d_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3, dim1, dim2, dim_s);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(mat3, d_mat3, size * sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_mat1));
        gpuErrchk(cudaFree(d_mat2));
        gpuErrchk(cudaFree(d_mat3));
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
    int dim_s = 4;
    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim2 = (int) atoi(argv[2]);

    int SIZE = dim1 * dim2;

    int* mat1 = (int*) malloc(dim1 * dim_s * sizeof(int));
    int* mat2 = (int*) malloc(dim_s * dim2 * sizeof(int));
    int* mat3 = (int*) malloc(SIZE * sizeof(int));

    initialize(mat1, dim1, dim_s);
    initialize(mat2, dim_s, dim2);

    matmul(mat1, mat2, mat3, dim1, dim2, dim_s);

    print(mat1, dim1, dim_s);
    print(mat2, dim_s, dim2);
    print(mat3, dim1, dim2);

    free(mat1);
    free(mat2);
    free(mat3);

    return 0;
}