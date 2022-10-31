#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "functions.h"


// size of graph
const size_t N = 10;

int main()
{
    // set N and pow to some big numbers and num_threads to 1 or 2 to see the boost in time 
    // omp_set_num_threads(8); 
    double start, end;
    // path lengths (power to which matrix will be raised)
    int pow = 3;
    
    // random adjacency matrix
    int *A = (int *) malloc(N * N * sizeof(int));
    RandomMatrix(&A[0], N);

    // result matrix
    int *res = (int *) malloc(N * N * sizeof(int));
    ZeroMatrix(&res[0], N);


    printf("----------------------------------\n");
    printf("Randomly generated adjacency matrix:\n");
    // PrintMatrix(A, N);

    start = omp_get_wtime();
    // #pragma omp parallel shared (A, res)
    MatPower(&A[0], &res[0], pow, N);
    end = omp_get_wtime();

    printf("\nResult paths of length = %d:\n", pow);
    // PrintMatrix(res, N);
    printf("----------------------------------\n");
    
    printf("Worked on %d threads, time elapsed = %f seconds\n", \
        omp_get_max_threads(), (end-start));

    free(A); 
    free(res);
    return 0;
}
