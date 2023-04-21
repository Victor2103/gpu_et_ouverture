// Import the necessary library for c programs. 
#include <stdlib.h>
#include <stdio.h>

/*
# Command to run in the terminal for cuda. It will create a conv2D.o file which is an executable. 
code=optimized_matmul && nvcc -o $code.o $code.cu && ./$code.o 
*/

// This function will handle error if you make a wrong configuration of the blocks
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
This function is the function who calcul the multiplication of 2 matrix with usage of GPU and thread with cuda. 
The 3 first arguments are the matrix. The first two matrix are the matrix to multiply between them and the last one is the result of the multiplication. 
The matrix are arguments for the device, They're are not the argument of the host. We allow some memories for the cuda device.
dim1 is the number of row of the first matrix and dim2 is the number of columns of th second matrix. 
dim_s is the number of columns of the first matrix and the number of row of the second matrix. It must be equal because if not, we can't do the multiplication. 
As a result the d_out matrix will be of dim1 * dim2.  
*/
__global__ void d_matmul(int* d_mat1, int* d_mat2, int* d_out, int dim1, int dim2, int dim_s)
{   
    
    // We define two shared variable to optimize the code. This variable are size 32*32 because the block dimension is 32. 
    // I can't put some variables to initialize the matrix I don't know why so I just put the number 32 directly. 
    // To create shared memory, it permits to read only global memory only once and then use the shared variable
    __shared__ int d_mat1Tmp[32][32], d_mat2Tmp[32][32];
    
    // We define 2 variables to have the row index (yIndex) and the column index (xIndex) define with the help of the threads and the blocks. 
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    // We verif if the index are not outsized. (it can happen with the dimension of your block)
    if (yIndex < dim1 && xIndex < dim2)
    {
        // For each index in the new matrix we will calcul his value and increment the sum value.
        int sum=0;
        
        /*We stock in the shared variable the value of each first matrix. 
        The threadIdx.x correspond to the id for the row of the thread. 
        The threadIdx.y correspond to the id of the columns of the thread. 
        And for each thread, we will make the for loop and replace the threadIdx.x or threadIdx.y by the value of k in the loop. 
        */
        d_mat1Tmp[threadIdx.y][threadIdx.x] = d_mat1[yIndex * dim_s + threadIdx.x];
        d_mat2Tmp[threadIdx.y][threadIdx.x] = d_mat2[threadIdx.y * dim2 + xIndex];

        // We synchronize the threads because the variable is shared. 
        __syncthreads();

        // We make the loop and we use the shared variable to increment the sum. 
        for (int k = 0;k < dim_s; k++) {
            sum += d_mat1Tmp[threadIdx.y][k] * d_mat2Tmp[k][threadIdx.x];
        }
        // We put the value of the sum at the good index inside the output matrix d_out.
        d_out[yIndex * dim2 + xIndex] = sum;  
    }
}

/*
This function will calcule the output matrix, result of the multiplication. It will be called when you don't have GPU on your machine. 
It is the basic function who can be implemented in c. 
We put the 3 matrices for the first argument, this matrix are not send to the device, there allocated with malloc and are pointer in c. 
The last arguments are :
dim1 (dim2) is the number of rows (columns) for the first (second) matrix. dim_s is the number of columns (rows) for the first (second) matrix. 
*/
void h_matmul(int* mat1, int* mat2, int* out, int dim1, int dim2, int dim_s) {
    // We go inside each index of the matrix with the two for loop
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            // We initialize the sum at 0.
            int sum = 0;
            // For each dimension of dim_s we increment the sum with the multiplication of matrix 1 and 2. 
            for (int k = 0; k < dim_s; k++) {
                sum += mat1[y * dim_s + k] * mat2[k * dim2 + x];
            }
            // We update the output value of the index with the sum incremented. 
            out[y * dim2 + x] = sum;
        }
    }
}

/* This function will calcule the matrix resulted by the multiplication of two matrix. 
If your device don't have GPU, the function will run the multiplication with cpu otherwise, we will use cuda.
We send some parameters like the 2 input matrix and the output matrix with no values. 
We also send the dimension of each matrix. mat1 (dim1,dim_s); mat2 (dim_s,dim2); mat3 (dim1,dim2)
This function doesn't return anything but will update the output matrix which is a pointer placed in the parameter. 
*/
void matmul(int* mat1, int* mat2, int* mat3, int dim1, int dim2,int dim_s) {
    // We check if we have some gpu and the number of gpu we have we the function cudaGetDeviceCount.
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    // If we don't have GPU, we run the function in c otherwise we initialize a cuda device.
    if (*deviceCount == 0) {
        h_matmul(mat1, mat2, mat3, dim1, dim2, dim_s);
    } else {
        // We define the variable if we have some GPU to make a configuration of our device. 
        // We define the dimension of each block. The maximum of ressources for one gpu is 32 for the dimension of the block so let's use it. 
        int size = dim1 * dim2;
        int BLOCK_DIM = 32;
        // With this dimension, we define the number of threads per block and the number of blocks per grid. 
        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 blocksPerGrid(
            (dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (dim2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (dim_s + threadsPerBlock.z - 1) / threadsPerBlock.z
        );
        // We print the threads per block and the block per grid to show our configuration on the terminal. 
        printf(
            "threadsPerBlock.x=%d, threadsPerBlock.y=%d, blocksPerGrid.x=%d, blocksPerGrid.y=%d\n",
            threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y
        );

        /* 
        We define three pointers.
        This pointers are defined for the device and not for the host. 
        Each pointer correspond to each matrix we need. The first to one are the initials matrix.
        The third one is the result matrix from the multiplication. 
        */
        int* d_mat1;
        int* d_mat2;
        int* d_mat3;

        // We allow for the 3 matrix some cuda memory for the device GPU.
        gpuErrchk(cudaMalloc((void**) &d_mat1, dim1 * dim_s * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat2, dim_s * dim2 * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat3, size * sizeof(int)));

        // We send the value of the matrix in the matrix of the device. mat1 to d_mat1 and mat2 to d_mat2. 
        // We just send the 2 initials matrix to be multiply because the result matrix will be filled and send after from the device to the host. 
        gpuErrchk(cudaMemcpy(d_mat1, mat1, dim1 * dim_s * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mat2, mat2, dim_s * dim2 * sizeof(int), cudaMemcpyHostToDevice));

        // We run the convolution function and specify all the blocks per grid and the threads per block inside the <<< >>>
        d_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3, dim1, dim2, dim_s);
        // We check if there are any error of configuration or other with cuda. 
        gpuErrchk(cudaPeekAtLastError());

        // Once the function have run, we send the result matrix from the device to the host. d_mat3 => mat3
        gpuErrchk(cudaMemcpy(mat3, d_mat3, size * sizeof(int), cudaMemcpyDeviceToHost));

        // We free the cuda memory of the 3 matrix. 
        gpuErrchk(cudaFree(d_mat1));
        gpuErrchk(cudaFree(d_mat2));
        gpuErrchk(cudaFree(d_mat3));
    }
    // We print the device count. If we have 0, we know we don't use cuda otherwise we have use cuda. 
    printf("deviceCount = %d\n", *deviceCount);
    free(deviceCount);
}

// With this function, we can initialize the matrix with values. For a position in the matrix, the value will be the position.
void initialize(int* mat, int dim1, int dim2) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            mat[y * dim2 + x] = y * dim2 + x;
        }
    }
}

// This function permits to print a row from a matrix. 
void printRow(int* row, int size) {
    printf("[");
    for (int x = 0; x < size - 1; x++) {
        printf("%d, ", row[x]);
    }
    if (size != 0) printf("%d", row[size - 1]);
    printf("]\n");
}

// This function permits to print a matrix in the console. 
void print(int* mat, int dim1, int dim2) {
    printf("[\n");
    for (int y = 0; y < dim1; y++) {
        printf("\t");
        printRow(&mat[y * dim2], dim2);
    }
    printf("]\n");
}

int main(int argc, char** argv) {
    /* We define 3 dimensions. for the multiplication of mat1 with mat2 and the result is mat3. 
    dim1 = dimension for rows first matrix mat1 and dimension for rows result matrix mat3 
    dim2 = dimension for columns second matrix mat2 and dimension for columns mat3
    dim_s = dimension for columns mat1 and dimension for rows mat2
    */
    int dim1 = 8;
    int dim2 = 8;
    int dim_s = 4;
    // If we want to enter the dimension of the matrix directly in the console, we can do it.
    // To do this, just put the number of rows and the number of columns you want after the command in the console. 
    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim2 = (int) atoi(argv[2]);

    // We define the size of the output matrix mat3.  
    int SIZE = dim1 * dim2;

    

    // We allocate some memory with malloc to the three matrix. 
    int* mat1 = (int*) malloc(dim1 * dim_s * sizeof(int));
    int* mat2 = (int*) malloc(dim_s * dim2 * sizeof(int));
    int* mat3 = (int*) malloc(SIZE * sizeof(int));

    // We initialize the 2 initial matrix to multiply.  
    initialize(mat1, dim1, dim_s);
    initialize(mat2, dim_s, dim2);

    // We call the multiplication function who will calculate our result matrix.
    matmul(mat1, mat2, mat3, dim1, dim2, dim_s);

    // We print the result of the three matrix, the 2 initials and the result of their multiplications.
    print(mat1, dim1, dim_s);
    print(mat2, dim_s, dim2);
    print(mat3, dim1, dim2);

    // We stop the allocation of memory of the three matrix. 
    free(mat1);
    free(mat2);
    free(mat3);

    return 0;
}