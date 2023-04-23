// Import the necessary library for c programs. 
#include <stdlib.h>
#include <stdio.h>

/*
# Command to run in the terminal for cuda. It will create a conv2D.o file which is an executable. 
code=conv2D && nvcc -o $code.o $code.cu && ./$code.o 
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
This function is the function who calcul the convolution matrix with usage of GPU and thread with cuda. 
The 3 first arguments are the matrix. The first one is the matrix initial, the second is the filter matrix and the last one is the result of the convolution. 
The matrix are arguments for the device, They're are not the argument of the host. We allow some memories for the cuda device.
The dim1 and dim2 are the dimension of the first matrix. It is the number of row and the number of columns. 
The dimFilter1 and dimFilter2 are the numbers of columns in the filter matrix for convolution
The outDim1 and outDim2 are the dimension for the output matrix.  
*/
__global__ void d_conv2D(int* d_mat1, int* d_mat2, int* d_out, int dim1, int dim2, int dimFilter1, int dimFilter2, int outDim1, int outDim2)
{
    // We define 2 variables to have the row index (yIndex) and the column index (xIndex) define with the help of the threads and the blocks. 
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int kCenterX = dimFilter1 / 2;
    int kCenterY = dimFilter1 / 2;
    // We verif if the index are not outsized.  (it can happen with the dimension of your block)
    if (yIndex < dim1 && xIndex < dim2)
    {
        // For each value of the new matrix we will calcul his convolution and increment the sum value. 
        int sum=0;
            for (int m = 0;m < dimFilter1; m++) {
                for (int n = 0; n < dimFilter2; n++){
                    int mm = dimFilter1 - 1 - m;
                    int nn = dimFilter1 - 1 - n;
                    int ii = yIndex + (m - kCenterY);
                    int jj = xIndex + (n - kCenterX);
                    if (ii >= 0 && ii < dim1 && jj >= 0 && jj < dim2) {
                        sum += d_mat1[ii * dim2 + jj] * d_mat2[mm * dimFilter1 + nn];
                }
            }
        }
        d_out[yIndex * dim2 + xIndex] = sum;
    }
}

__global__ void d_optimized_conv2D(int* d_mat1, int* d_mat2, int* d_out, int dim1, int dim2, int dimFilter1, int dimFilter2, int outDim1, int outDim2)
{
    // We define 2 variables to have the row index (yIndex) and the column index (xIndex) define with the help of the threads and the blocks. 
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int kCenterX = dimFilter1 / 2;
    int kCenterY = dimFilter1 / 2;

    // We define two shared variable to optimized the time. The first one is for the intial matrix and the second one is for the convolution matrix. 
    // We use two dimension because the matrix are in 2D. 
    // We make a size 32*32 because it is the block size. DLOCK_DIM=32. 
    __shared__ int d_mat1Tmp[32][32], d_mat2Tmp[32][32];

    

    // We verif if the index are not outsized.  (it can happen with the dimension of your block)
    if (yIndex < outDim1 && xIndex < outDim2)
    {
        // For each value of the new matrix we will calcul his convolution and increment the sum value. 
        int sum=0;

        // We initialize this shared variable with the help of the threadIdx.x and threadIdx.y
        d_mat1Tmp[threadIdx.y][threadIdx.x] = d_mat1[threadIdx.y * dim2 + threadIdx.x];
        d_mat2Tmp[threadIdx.y][threadIdx.x] = d_mat2[threadIdx.y * dimFilter2 + threadIdx.x];

        // We synchronize the threads because the variable is shared. 
        __syncthreads();
        
        for (int m = 0;m < dimFilter1; m++) {
            for (int n = 0; n < dimFilter2; n++){
                int mm = dimFilter1 - 1 - m;
                int nn = dimFilter1 - 1 - n;
                int ii = yIndex + (m - kCenterY);
                int jj = xIndex + (n - kCenterX);
                if (ii >= 0 && ii < dim1 && jj >= 0 && jj < dim2) {
                    sum += d_mat1Tmp[ii][jj] * d_mat2Tmp[mm][nn];
                }
            }
        }
        d_out[yIndex * dim2 + xIndex] = sum;
    }
}

/*
This function permits to define the output dimension of the result matrix. 
We put in paramaters the table with the two output dimension. This table is a pointer and will be define in the function. 
The 4 following parameters are the dimension of the input matrix and the dimension of the filter matrix. 
*/
void setOutDims(int* outDims, int matDim1, int matDim2, int filterDim1, int filterDim2) {
    outDims[0] = matDim1 - filterDim1 + 1;
    outDims[1] = matDim2 - filterDim2 + 1 ;
}


/*
This function will calcule the output matrix. It will be called when you don't have GPU on your machine. 
It is the basic function who can be implemented in c. 
We put the 3 matrices for the first argument, this matrix are not send to the device, there allocated with malloc and are pointer in c. 
The last arguments are the dimension of the intput matrix and the output matrix
*/
void h_conv2D(int *mat1, int *mat3, int dim1, int dim2, int *mat2, int filterDim1,int outDim1) {
    int i, j, m, n;
    int kCenterX = filterDim1 / 2;
    int kCenterY = filterDim1 / 2;
    float sum;
    for (i = 0; i < dim2; ++i) {
        for (j = 0; j < dim1; ++j) {
            sum = 0;
            for (m = 0; m < filterDim1; ++m) {
                for (n = 0; n < filterDim1; ++n) {
                    int mm = filterDim1 - 1 - m;
                    int nn = filterDim1 - 1 - n;
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);
                    if (ii >= 0 && ii < dim2 && jj >= 0 && jj < dim1) {
                        sum += mat1[ii * dim1 + jj] * mat2[mm * filterDim1 + nn];
                    }
                }
            }
            if (i * dim1 + j < outDim1 * outDim1){
                mat3[i * dim1 + j] = sum;
            }
        }
    }
}




/* This function will calcule the matrix resulted by the convolution. 
If your device don't have GPU, the function will run the convolution with cpu otherwise, we will use cuda.
We send some parameters like the input matrix, the convolution matrix, the output matrix with no values. 
We also send the dimension of each matrix. 
This function doesn't return anything but will update the output matrix which is a pointer placed in the parameter. 
*/
void conv2D(int* mat1, int* mat2, int* mat3, int dim1, int dim2, int dimFilter1, int dimFilter2, int outDim1,int outDim2) {
    // We check if we have some gpu and the number of gpu we have we the function cudaGetDeviceCount. 
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
    // If we don't have GPU, we run the function in c otherwise we initialize a cuda device. 
    if (*deviceCount == 0) {
        h_conv2D(mat1, mat3, dim1, dim2, mat2, dimFilter1, outDim1);
    } else {
        // We define the variable if we have some GPU to make a configuration of our device. 
        // We define the dimension of each block. The maximum of ressources for one gpu is 32 for the dimension of the block so let's use it. 
        int BLOCK_DIM = 32;
        // With this dimension, we define the number of threads per block and the number of blocks per grid. 
        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 blocksPerGrid(
            (dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (dim2 + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        // We print the threads per block and the block per grid to show our configuration on the terminal. 
        printf(
            "threadsPerBlock.x=%d, threadsPerBlock.y=%d, blocksPerGrid.x=%d, blocksPerGrid.y=%d\n",
            threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y
        );

        /* 
        We define three pointers.
        This pointers are defined for the device and not for the host. 
        Each pointer correspond to each matrix we need. The first one is the initial matrix.
        The second is the convolution matrix and the third one is the result matrix. 
        */
        int* d_mat1;
        int* d_mat2;
        int* d_mat3;

        // We initialiaze variable to count the execution time of each function. 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds;

        // We allow for the 3 matrix some cuda memory for the device GPU. 
        gpuErrchk(cudaMalloc((void**) &d_mat1, dim1 * dim2 * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat2, dimFilter1 * dimFilter2 * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat3, outDim1 * outDim2 * sizeof(int)));
        
        // We send the value of the matrix in the matrix of the device. mat1 to d_mat1 and mat2 to d_mat2. 
        // We just send the initial matrix and the convolution matrix because the result matrix will be filled and send after from the device to the host. 
        gpuErrchk(cudaMemcpy(d_mat1, mat1, dim1 * dim2 * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mat2, mat2, dimFilter1 * dimFilter2 * sizeof(int), cudaMemcpyHostToDevice));

        // We start an event to evaluate the naive function for multiplication of matrix with GPU. 
        gpuErrchk(cudaEventRecord(start));

        // We run the convolution function and specify all the blocks per grid and the threads per block inside the <<< >>>
        d_conv2D<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3, dim1, dim2, dimFilter1, dimFilter2, outDim1, outDim2);
        // We check if there are any error of configuration or other with cuda. 
        gpuErrchk(cudaPeekAtLastError());

        // We stop the event and print to the user the time elasped during the function. 
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("convolution 2D naive function GPU milliseconds = %fms\n", milliseconds);

        // We start an event to evaluate the optimized function for multiplication of matrix with GPU. 
        gpuErrchk(cudaEventRecord(start));

        // We run the convolution function and specify all the blocks per grid and the threads per block inside the <<< >>>
        d_optimized_conv2D<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3, dim1, dim2, dimFilter1, dimFilter2, outDim1, outDim2);
        // We check if there are any error of configuration or other with cuda. 
        gpuErrchk(cudaPeekAtLastError());

        // We stop the event and print to the user the time elasped during the function. 
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("convolution 2D optimized function GPU milliseconds = %fms\n", milliseconds);

        // We start an event to evaluate the function for multiplication of matrix with CPU (no GPU on the device). 
        gpuErrchk(cudaEventRecord(start));

        // We run the basic function in C
        h_conv2D(mat1, mat3, dim1, dim2, mat2, dimFilter1, outDim1);
        // We check if there are any error of configuration or other with cuda. 
        gpuErrchk(cudaPeekAtLastError());

        // We stop the event and print to the user the time elasped during the function. 
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("convolution 2D function no GPU milliseconds = %fms\n", milliseconds);

        // Once the function have run, we send the result matrix from the device to the host. d_mat3 => mat3
        gpuErrchk(cudaMemcpy(mat3, d_mat3, outDim1 * outDim2 * sizeof(int), cudaMemcpyDeviceToHost));

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
    // We define 4 dimensions, 2 for the matrix initial and 2 for the convolution matrix. 
    int dim1 = 30;
    int dim2 = 30;
    int filterDim1 = 3 ;
    int filterDim2 = 3 ;
    // If we want to enter the dimension of the matrix directly in the console, we can do it.
    // To do this, just put the number of rows and the number of columns you want after the command in the console. 
    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim2 = (int) atoi(argv[2]);

    // With the function setOutDims, we set up the dimension of our result matrix and save it in two integers. 
    int* outDims = (int*) malloc(2 * sizeof(int));
    setOutDims(outDims, dim1, dim2, filterDim1, filterDim2);
    int outDim1 = outDims[0];
    int outDim2 = outDims[1];
    free(outDims);

    // We allocate some memory with malloc to the three matrix. 
    int* mat1 = (int*) malloc(dim1 * dim2 * sizeof(int));
    int* mat2 = (int*) malloc(filterDim1 * filterDim2 * sizeof(int));
    int* mat3 = (int*) malloc(outDim1 * outDim2 * sizeof(int));

    // We initialize the initial matrix and the convolution matrix with values. 
    initialize(mat1, dim1, dim2);
    initialize(mat2, filterDim1, filterDim2);

    // We call the convolution function who will calculate our third matrix.
    conv2D(mat1, mat2, mat3, dim1, dim2, filterDim1, filterDim2, outDim1, outDim2);


    // We print the result of the three matrix, the initial, the convolution and then the result matrix. 
    // By default, it put it in comments because matrix has 30*30 size. 
    /*
    print(mat1, dim1, dim2);
    print(mat2, filterDim1, filterDim2);
    print(mat3, outDim1, outDim2);
    */


    // We stop the allocation of memory of the three matrix. 
    free(mat1);
    free(mat2);
    free(mat3);

    return 0;
}