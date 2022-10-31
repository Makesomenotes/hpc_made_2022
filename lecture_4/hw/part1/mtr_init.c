#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "functions.h"


void ZeroMatrix(int * A, size_t N)
{
    #pragma omp parallel for shared(A)
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<N; j++)
        {
            A[i * N + j] = 0;
        }
    }
}

void RandomMatrix(int * A, size_t N)
{
    srand(time(NULL));
    // double temp;

    #pragma omp parallel for shared(A)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // temp = (double)rand() / (double)RAND_MAX;
            // A[i * N + j] = (int) (temp*10.);

            // binary case
            A[i * N + j] = (int) (rand() % 2); 
        }
    }
}
