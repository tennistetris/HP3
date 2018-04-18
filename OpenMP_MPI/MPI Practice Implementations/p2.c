/* Example: simple data type
 *
 * Send and Receive an Integer Data
 * Send and Receive an Array of Doubles
*/ 
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MSIZE 10

int main(int argc, char *argv[]) {

    MPI_Status status;
    int rank, nprocs;

    int i, j;

    /* data to communicate */
    int     data_int;
    double  data_double;
    double  matrix[MSIZE][MSIZE];
    

    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
    /* simple datatype communication examples using two processes */
    if (nprocs < 2) {

        if (rank == 0) {
            printf("\nSORRY: need at least 2 processes.\n");
        }
        /* exit MPI on error */
        MPI_Finalize();
        
        exit(EXIT_FAILURE);
    }
    
    /* INTEGER TYPE */
    if (rank == 0) {

        data_int = 10;

        MPI_Send(&data_int, 1, MPI_INT, 1, 666, MPI_COMM_WORLD); 
    }

    if (rank == 1) {
        
        MPI_Recv(&data_int, 1, MPI_INT, 0, 666, MPI_COMM_WORLD, &status);

        printf("\nProc 1 receives %d from proc 0.\n", data_int);
    }
    
    /* ARRAY OF DOUBLE TYPE */
    if (rank == 0) {

        for (i = 0; i < MSIZE; i++)
            for (j = 0; j < MSIZE; j++)
                matrix[i][j] = (double) i+j;

        MPI_Send(matrix, MSIZE*MSIZE, MPI_DOUBLE, 1, 666, MPI_COMM_WORLD); 
    
    }
    if (rank == 1) {

        MPI_Recv(matrix, MSIZE*MSIZE, MPI_DOUBLE, 0, 666, MPI_COMM_WORLD, &status);

        printf("\nProc 1 receives the following matrix from proc 0.\n");
        for (i = 0; i < MSIZE; i++) {
            for (j = 0; j < MSIZE; j++) {
                printf("%6.2f ", matrix[i][j]);
            }
            printf("\n");
        }

    }

    /* Quit */
    MPI_Finalize();

    return 0;
}
