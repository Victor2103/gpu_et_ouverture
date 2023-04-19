/*
Add two tables of 1D beetween them element by element
command to run : gcc -Wall -Werror -o somme.out somme.c && ./somme.out */

#include <stdio.h>
#include <stdlib.h>

void afficherTableau(int *tableau, int sizeTableau)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        printf("%d ", tableau[i]);
    }
    printf("\n");
}

void addTableau(int *tab1, int *tab2, int *sommeTab, int tabSize)
{
    int i = 0;
    for (i = 0; i < tabSize; i++)
    {
        sommeTab[i] = tab1[i] + tab2[i];
    }
}

int main(int argc, char *argv[])
{
    int i = 0, SIZE = 20;
    int *tableau1 = malloc(SIZE * sizeof(int));
    int *tableau2 = malloc(SIZE * sizeof(int));
    int *sommeTableau = malloc(SIZE * sizeof(int));
    for (i = 0; i < SIZE; i++)
    {
        tableau1[i] = i;
        tableau2[i] = i;
    }
    addTableau(tableau1, tableau2, sommeTableau, SIZE);
    afficherTableau(tableau1, SIZE);
    afficherTableau(tableau2, SIZE);
    afficherTableau(sommeTableau, SIZE);
    free(tableau1);
    free(tableau2);
    free(sommeTableau);
    return (0);
}