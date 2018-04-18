/*
Used MPI_Scatter and MPI_Sendrecv
*/

//NOTE:- Output file done with thousand elements since 10000X10000 would take a lot of memory

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>


int PrintMatrix (char *wheretoprint, float* matrix, 
                 int nrows, int ncols);

int main(int argc, char *argv[])
{
    const int  MSIZE=4;
    int rank, nprocs;

    int i, j;

    int localsize;
    float * matrix;
    float *matrix_transpose;
    matrix = (float*)malloc(MSIZE*MSIZE*sizeof(float));
    

    
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    /* Check that we have an even number of processes */
    if (MSIZE % nprocs != 0) {
        printf("\nYou have to use a number of processes that fit %d\n", MSIZE);

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {    /* Process 0 read matrix*/
        for(int i = 0;i<MSIZE;i++){
            for(int j = 0;j<MSIZE;j++){
                matrix[i*MSIZE + j]= i*j/(float)MSIZE;
            }
        }

    }
    //if(rank == 0)
    //    PrintMatrix("matrixTranspose.txt",matrix,MSIZE,MSIZE);


    /* allocate local structures*/
    float *local_matrix;
    localsize = MSIZE/nprocs;
    local_matrix = (float*)malloc(localsize*MSIZE*sizeof(float));

    
    // Scattering matrix to all proceses, place it in local_matrix
    MPI_Scatter (matrix, localsize * MSIZE, MPI_FLOAT,   // sender info
        local_matrix, localsize * MSIZE, MPI_FLOAT,   // receivers info
        0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    /*if(rank == 0){
        free(matrix);
        matrix_transpose = (float*)malloc(MSIZE*MSIZE*sizeof(float));
    }*/

    //PrintMatrix("matrixTranspose.txt",local_matrix,localsize,MSIZE);
    

    MPI_Status * status;
    int ftag = 101;
    for(int i = 0;i < localsize;i++ ){
        for(int j = 0;j<MSIZE;j++){
            MPI_Sendrecv(&local_matrix[i*MSIZE + j],1,MPI_FLOAT,0,ftag,&matrix[(j*MSIZE + i + rank*localsize)],1,MPI_FLOAT,rank,ftag,MPI_COMM_WORLD,status);
        }
    }
    
    if(rank == 0)
        PrintMatrix("matrixTranspose.txt",matrix,MSIZE,MSIZE);

    /*
    free(local_matrix);
    MPI_Barrier(MPI_COMM_WORLD);
    
    char OUTFILE[] = "matrixTranspose.txt"; 
    PrintMatrix (OUTFILE, matrix_transpose, MSIZE, MSIZE);
    
    printf("\nProcess %d is gathering matrix from all %d processes\n",
        rank, nprocs);

    printf("\nResulst can be find in file %s\n\n", OUTFILE);
    */


    /* Quit */
    MPI_Finalize();

    return 0;
}



int PrintMatrix (char *wheretoprint, float *matrix, int nrows, int ncols)
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
            fprintf(fp, "%f ", matrix[i*ncols + j]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n");

    fclose(fp);

    return 0;
}
