#include <stdio.h>
#include <stdlib.h>

/*
We suppose the tab has 20 of length
Command to run : gcc -Wall -Werror -o transpose.out transpose.c && ./transpose.out
*/

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


void init2DMatrix (int *tableau,int sizeRow,int sizeColumns){
    int i = 0;
    for (i = 0; i < sizeRow*sizeColumns; i++){
        tableau[i]=i;
    }
}


void transpose(int *tableau, int *tableauTranspose, int sizeRow, int sizeColumns)
{
}

int main(int argc, char *argv[])
{
    int SIZEROW = 5, SIZECOLUMNS = 10;
    int *matrix = malloc(SIZEROW * SIZECOLUMNS * sizeof(int));
    init2DMatrix(matrix,SIZEROW,SIZECOLUMNS);
    afficher2DMatrix(matrix, SIZEROW, SIZECOLUMNS);
    return (0);
}