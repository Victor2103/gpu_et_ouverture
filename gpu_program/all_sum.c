#include <stdio.h>
#include <stdlib.h>

/*
We suppose the tab has 20 of length
Command to run : gcc -Wall -Werror -o all_sum.out all_sum.c && ./all_sum.out
*/

void afficherTableau(int *tableau, int sizeTableau)
{
    int i = 0;
    for (i = 0; i < sizeTableau; i++)
    {
        printf("%d ", tableau[i]);
    }
    printf("\n");
}

int all_sum(int *tableau, int total, int sizeTab)
{
    int i = 0;
    for (i = 0; i < sizeTab; i++)
    {
        total = total + tableau[i];
    }
    return (total);
}

int main(int argc, char *argv[])
{
    int i = 0, SIZE = 20;
    int *tableauInit = malloc(SIZE * sizeof(int));
    for (i = 0; i < SIZE; i++)
    {
        tableauInit[i] = i;
    }
    int total = all_sum(tableauInit, 0, SIZE);
    printf("Voici le total : %d \n", total);
    free(tableauInit);
    return (0);
}