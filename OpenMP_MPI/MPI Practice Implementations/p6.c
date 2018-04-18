/* Exercise: Matrix * Vector
 * 
 * In this exercise you will compute the matrix vector 
 * product of a matrix A with a vector V of double.
 * 
 * Matrix A is split among processes. Partial results are 
 * collected to master process and written to file. 
 *
 * MPI SCATTER and GATHER functions will be used
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define MSIZE 8
#define OUTFILE "matrixvec.out"

/* Auxiliary function to print content of integer matrix */
int PrintMatrix (char *wheretoprint, int *integermatrix, 
                 int nrows, int ncols);

int main(int argc, char *argv[])
{
    int rank, nprocs;

    int i, j;

    /* data */
    int localsize;
    int matrix[MSIZE*MSIZE], vector[MSIZE];
    int *local_matrix, *local_vector;
    
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    /* print banner */
    if (rank == 0) {
        printf("\nMatrix-vector product example: matrix A is split among\n");
        printf("processes and a local matrix-vector product is computed.\n");
        printf("Partial results are collected to master process and\n");
        printf("written to file. Works only with matrix size commensurable\n");
        printf("with the number of involved processes.\n\n");

        fflush(stdout);
    }

    /* Check that we have an even number of processes */
    if (MSIZE % nprocs != 0) {
        printf("\nYou have to use a number of processes that fit %d\n", MSIZE);

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {    /* Process 0 initialize matrix */

        /* Preparing data structures */
        for (i = 0; i < MSIZE; i++)
            for (j = 0; j < MSIZE; j++)
                matrix[i*MSIZE+j] = i+j+2;

        for (j = 0; j < MSIZE; j++)
            vector[j] = j+1;

        /* print input data */
        PrintMatrix (OUTFILE, matrix, MSIZE, MSIZE);
        PrintMatrix (OUTFILE, vector, 1, MSIZE);

        printf("\nProcess %d is distributing matrix to all %d processes\n",
            rank, nprocs);

    }

    /* allocate local structures and init to zero */
    localsize = MSIZE / nprocs;
    local_matrix = (int *) calloc (localsize * MSIZE , sizeof(int));
    local_vector = (int *) calloc (localsize , sizeof(int));

    /* Scattering matrix to all proceses, place it in local_matrix */
    MPI_Scatter (matrix, localsize * MSIZE, MPI_INT,    /* sender info */
        local_matrix, localsize * MSIZE, MPI_INT,   /* receivers info */
        0, MPI_COMM_WORLD);

    /* main process broadcast vector V */
    MPI_Bcast(&vector, MSIZE, MPI_INT, 0, MPI_COMM_WORLD);

    /* Begin of product matrix * vector */
    for (i = 0; i < localsize; i++)
        for (j = 0; j < MSIZE; j++)
            local_vector[i] += local_matrix[i*MSIZE+j] * vector[j];  


    /* Gathering local_vectors from all proceses, place them in vector */
    MPI_Gather (local_vector, localsize, MPI_INT,    /* senders info */
                vector, localsize, MPI_INT,          /* receiver info */
                0, MPI_COMM_WORLD);


    if (rank == 0) {

        /* print input data */
        PrintMatrix (OUTFILE, vector, 1, MSIZE);
        
        printf("\nProcess %d is gathering matrix from all %d processes\n",
            rank, nprocs);

        printf("\nResulst can be find in file %s\n\n", OUTFILE);

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
