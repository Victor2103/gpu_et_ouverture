#include <stdio.h>
#include <stdlib.h>

/*
We suppose the tab has 20 of length
Command to run : gcc -Wall -Werror -o decalage.out decalage.c && ./decalage.out
*/

void decalage(int *tableau, int sizeTableau, int *tableauDecale)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        if (i == 0)
        {
            tableauDecale[0] = tableau[sizeTableau - 1];
        }
        else
        {
            tableauDecale[i] = tableau[i - 1];
        }
    }
}

void afficherTableau(int *tableau, int sizeTableau)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        printf("%d ", tableau[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int i = 0, SIZE = 20;
    int *tableauDecale = malloc(SIZE * sizeof(int));
    int *tableauInit = malloc(SIZE * sizeof(int));
    for (i = 0; i < SIZE; i++)
    {
        tableauInit[i] = i;
    }
    decalage(tableauInit, SIZE, tableauDecale);
    afficherTableau(tableauInit, SIZE);
    afficherTableau(tableauDecale, SIZE);
    free(tableauDecale);
    free(tableauInit);
    return (0);
}