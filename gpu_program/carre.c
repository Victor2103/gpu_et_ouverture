#include <stdio.h>
#include <stdlib.h>

/*
We suppose the tab has 20 of length
Command to run : gcc -Wall -Werror -o carre.out carre.c && ./carre.out
*/

void carre(int *tableau, int sizeTableau, int *tableauCarre)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        tableauCarre[i] = tableau[i] * tableau[i];
    }
}

void afficherTableau(int *tableau, int sizeTableau)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        printf("%d ", tableau[i]);
    }
}

int main(int argc, char *argv[])
{
    int i = 0, SIZE = 20;
    int *tableauCarre = malloc(SIZE * sizeof(int));
    int *tableauInit = malloc(SIZE * sizeof(int));
    for (i = 0; i < SIZE; i++)
    {
        tableauInit[i] = i;
    }
    carre(tableauInit, SIZE, tableauCarre);
    afficherTableau(tableauCarre, SIZE);
    free(tableauCarre);
    free(tableauInit);
    return (0);
}