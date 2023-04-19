#include <stdio.h>
#include <stdlib.h>

// gcc conv2D.c -Wall -Werror -o conv2D.o && ./conv2D.o

void matmul(int* mat1, int* mat2, int* out, int dim1, int dim2, int dim_s) {
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

void setOutDims(int* outDims, int matDim1, int matDim2, int filterDim1, int filterDim2) {
    int l1 = filterDim1 / 2;
    int l2 = filterDim2 / 2;
    outDims[0] = matDim1 - 2 * l1;
    outDims[1] = matDim2 - 2 * l2;
}

void conv2D(int* mat, int* filter, int* out, int matDim1, int matDim2, int filterDim1, int filterDim2) {
    int* outDims = (int*) malloc(2 * sizeof(int));
    setOutDims(outDims, matDim1, matDim2, filterDim1, filterDim2);
    int outDim1 = outDims[0];
    int outDim2 = outDims[1];
    free(outDims);
    for (int y = 0; y < outDim1; y++) {
        for (int x = 0; x < outDim2; x++) {
            int o = 0;
            for (int j = 0; j < filterDim1; j++) {
                for (int i = 0; i < filterDim2; i++) {
                    o += mat[(y + j) * matDim2 + x + i] * filter[j * filterDim2 + i];
                }
            }
            out[y * outDim2 + x] = o;
        }
    }
}

void initialize(int* mat, int dim1, int dim2) {
    for (int x = 0; x < dim2; x++) {
        for (int y = 0; y < dim1; y++) {
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
    int matDim1 = 6;
    int matDim2 = 4;
    int filterDim1 = 3;
    int filterDim2 = 3;
    if (argc > 1) matDim1 = (int) atoi(argv[1]);
    if (argc > 2) matDim2 = (int) atoi(argv[2]);
    if (argc > 3) filterDim1 = (int) atoi(argv[3]);
    if (argc > 4) filterDim2 = (int) atoi(argv[4]);

    int* outDims = (int*) malloc(2 * sizeof(int));
    setOutDims(outDims, matDim1, matDim2, filterDim1, filterDim2);
    int outDim1 = outDims[0];
    int outDim2 = outDims[1];
    free(outDims);

    int MAT_SIZE = matDim1 * matDim2;
    int FILTER_SIZE = filterDim1 * filterDim2;
    int OUT_SIZE = outDim1 * outDim2;

    int* mat = (int*) malloc(MAT_SIZE * sizeof(int));
    int* filter = (int*) malloc(FILTER_SIZE * sizeof(int));
    int* out = (int*) malloc(OUT_SIZE * sizeof(int));

    initialize(mat, matDim1, matDim2);
    initialize(filter, filterDim1, filterDim2);

    conv2D(mat, filter, out, matDim1, matDim2, filterDim1, filterDim2);

    print(mat, matDim1, matDim2);
    print(filter, filterDim1, filterDim2);
    print(out, outDim1, outDim2);

    free(mat);
    free(filter);
    free(out);

    return 0;
}