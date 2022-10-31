#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "functions.h"


void PrintMatrix(int * A, size_t N)
{
    for (int i = 0; i < N; i++)
    {
     for (int j = 0; j < N; j++)
         printf("%d\t", A[N * i + j]);
     printf("\n");
    }
}


void MatMul(int * A, int * B, int * res, size_t N)
{

    // make temporal matrix to store result during computations
    int *temp = (int *) malloc(N * N * sizeof(int));
    ZeroMatrix(&temp[0], N);
    
    #pragma omp parallel for shared(temp, A, B)
    for (int k = 0; k < N; k++)
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
                temp[i * N + j] = temp[i * N + j] + A[i * N + k] * B[k * N + j];
        }

    // write to resulting matrix
    #pragma omp parallel for shared(temp, res)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            res[i * N + j] = temp[i * N + j];
}

void MatPower(int * A, int * res, int pow, size_t N)
{
    // base condition
    if (pow == 1) {
        #pragma omp parallel for shared(A, res)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                res[i * N + j] = A[i * N + j];
            }
        return;
    }
 
    // recursion call for first half
    MatPower(A, res, pow / 2, N);
 
    // multiply two halves
    MatMul(res, res, res, N);
 
    // if pow is odd
    if (pow % 2 != 0)
        MatMul(res, A, res, N);
}
