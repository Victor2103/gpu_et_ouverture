/*
Add two tables of 1D beetween them element by element
command to run : gcc -Wall -Werror -o multiplication.out multiplication.c && ./multiplication.out */

#include <stdio.h>
#include <stdlib.h>

void afficher2DMatrix(int *tableau, int sizeRow, int sizeColumns)
{
    int i = 0, j = 0;
    for (i = 0; i < sizeRow; i++)
    {
        for (j = 0; j < sizeColumns; j++)
        {
            printf("%d ", tableau[sizeColumns * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init2DMatrix(int *tableau, int sizeRow, int sizeColumns)
{
    int i = 0;
    for (i = 0; i < sizeRow * sizeColumns; i++)
    {
        tableau[i] = i;
    }
}

void multiply2Matrix(int *matrice1, int *matrice2, int *produitMatrice, int sizeRow, int sizeColumns)
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < sizeRow; i++)
    {
        for (j = 0; j < sizeColumns; j++)
        {
            produitMatrice[sizeColumns * i + j] = 0;
            for (k = 0; k < sizeRow; k++)
            {
                produitMatrice[sizeColumns * i + j] = produitMatrice[sizeColumns * i + j] + matrice1[sizeColumns * i + k] + matrice2[sizeRow * i + k];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int SIZEROW1 = 5, SIZECOLUMNS1 = 5, SIZEROW2 = 5, SIZECOLUMNS2 = 5;
    int *matrix1 = malloc(SIZEROW1 * SIZECOLUMNS1 * sizeof(int));
    int *matrix2 = malloc(SIZEROW2 * SIZECOLUMNS2 * sizeof(int));
    int *produitMatrix = malloc(SIZEROW1 * SIZECOLUMNS2 * sizeof(int));
    init2DMatrix(matrix1, SIZEROW1, SIZECOLUMNS1);
    init2DMatrix(matrix2, SIZEROW2, SIZECOLUMNS2);
    multiply2Matrix(matrix1, matrix2, produitMatrix, SIZEROW1, SIZECOLUMNS2);
    afficher2DMatrix(produitMatrix, SIZEROW1, SIZECOLUMNS2);
    free(matrix1);
    free(matrix2);
    free(produitMatrix);
    return (0);
}