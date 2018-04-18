/* Exercise: Matrix * Matrix
 * 
 * In this exercise you will compute the matrix vector 
 * product of two matrix A and B.
 * 
 * Matrix B is split among processes. Partial results are 
 * collected to master process and written to file. 
 *
 * MPI SCATTER and GATHER functions will be used
 *
 * This version works only with matrix size commensurable
 * with the number of involved processes.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define OUTFILE "matrix.out"

/* Auxiliary function to print content of integer matrix */
int PrintMatrix (char *wheretoprint, int *integermatrix, 
                 int nrows, int ncols);

int main(int argc, char *argv[])
{
    int rank, nprocs;

    int i, j, k;

    /* data */
    int msize, *matrixA, *matrixB, *matrixC;
    int *local_matrixA, *local_matrixC;
    int localsize, remain;
    
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if (rank == 0) {
        /* print banner */
        printf("\nMatrix-matrix product example: matrix A is split among\n");
        printf("processes while matrix B is broadcasted. Local product\n");
        printf("is computed and partial results are collected to master\n");
        printf("process and written to file.\n\n");

        fflush(stdout);
    
        /* If we are the "console" process, get a integer from the user to
        specify the dimension of matrix */
        printf("Enter the matrix size to compute: \n");
        fflush(stdout);

        scanf("%d", &msize);

        if (msize <= 0) /* at least one */
            msize = nprocs;
    
        printf("Matrix size set to %d.\n", msize);
        fflush(stdout);
    }

    /* main process broadcast matrix size */
    MPI_Bcast(&msize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* must be divisible */
    if (msize%nprocs != 0) {
        if (rank == 0) 
            printf("You have to use a number of process that fit %d\n",msize);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    /* all processes allocate matrix B */
    matrixB = (int *) malloc (msize*msize * sizeof(int));
    if (matrixB == NULL) {
        printf("Cannot allocate matrix B.\n");
    }

    /* master allocate matrix A and C */
    if (rank == 0) {
        matrixA = (int *) malloc (msize*msize * sizeof(int));
        if (matrixA == NULL) {
            printf("Cannot allocate matrix A.\n");
        }

        matrixC = (int *) malloc (msize*msize * sizeof(int));
        if (matrixC == NULL) {
            printf("Cannot allocate matrix C.\n");
        }

        /* initialize matrix A and B */
        for (i = 0; i < msize; i++) {
            for (j = 0; j < msize; j++) {
                matrixA[i*msize+j] = i+j+2;
                matrixB[i*msize+j] = i+j+2;
            }
        }

        if (msize <= 16) {
            /* print input data */
            PrintMatrix(OUTFILE, matrixA, msize, msize);
            PrintMatrix(OUTFILE, matrixB, msize, msize);
        }
    }


    /* compute local size */
    localsize = msize / nprocs;

    /* allocate local structures */
    local_matrixA = (int *) malloc (localsize * msize * sizeof(int));
    local_matrixC = (int *) malloc (localsize * msize * sizeof(int));

    /* main process broadcast matrix B */
    MPI_Bcast(matrixB, msize*msize, MPI_INT, 0, MPI_COMM_WORLD);

    /* Scattering matrix A all proceses */
    MPI_Scatter (matrixA, localsize*msize, MPI_INT,     /* sender info */
            local_matrixA, localsize*msize, MPI_INT,    /* receivers info */
            0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing  matrix * matrix product ...");
    }
    /* Begin of product matrix * submatrix */
    for (i = 0; i < localsize; i++) {
        for (j = 0; j < msize; j++) {

            local_matrixC[i*msize+j] = 0.0;

            for (k = 0; k < msize; k++) {
                local_matrixC[i*msize+j] += local_matrixA[i*msize+k] * 
                                    matrixB[k*msize+j];  
            }
        }
    }
    
    /* Gathering of matrix C from all proceses */
    MPI_Gather(local_matrixC, localsize*msize, MPI_INT, /* senders info */
                matrixC, localsize*msize, MPI_INT,      /* receiver info */
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("done.\n\n");

        if (msize <= 16)
        PrintMatrix(OUTFILE, matrixC, msize, msize);
    }


    /* Quit */
    MPI_Finalize();

    return 0;
}



int PrintMatrix (char *wheretoprint, int *integermatrix, int nrows, int ncols)
{
    FILE *fp;

    int i, j;

    fp = fopen(wheretoprint,"a");
    if (fp == NULL) {
        printf("\nCannot write on %s\n", wheretoprint);
        return -1;
    }

    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++) {
            fprintf(fp, "%5d ", integermatrix[i*ncols + j]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n");

    fclose(fp);

    return 0;
}
