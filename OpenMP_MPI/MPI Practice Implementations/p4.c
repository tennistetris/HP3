/* Exercise: Circular Shift
 *
 * In this exercise you will communicate a matrix 
 * among process in a circular topology, so that
 * every process has a left and a right neighbor
 * to receive and to send data respectivelly.
 *
 * Check what happens for huge size of the message
 */
#include <stdio.h>
#include <mpi.h>

/* MSIZE is the size of the integer arrays to send */
/* try to set MSIZE to 500, 5000 and 50000 */
/* does it work with all these sizes? What happen ? */
#define MSIZE 500

int main(int argc, char *argv[])
{
    MPI_Status status;
    int rank, size, tag, to, from;

    int i;
    int A[MSIZE], B[MSIZE];
    
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* print banner */
    if (rank == 0) {
        printf("\nCircular shift example: each process fills matrix A\n");
        printf("elements with its rank and sends them to the right\n");
        printf("process which receives them into matrix B.\n\n");

        fflush(stdout);
    }

    /* Arbitrarily choose 201 to be our tag.  Calculate the rank of the
       to process in the ring.  Use the modulus operator so that the
       last process "wraps around" to rank zero. */
    
    tag = 201;
    to = (rank + 1) % size;
    from = (rank + size - 1) % size;

    for (i = 0; i < MSIZE; i++)
        A[i] = rank;

    /* starting send of array A */
    MPI_Send(A, MSIZE, MPI_INT, to, tag, MPI_COMM_WORLD); 
    printf("Proc %d sends %d integers to proc %d\n", 
        rank, MSIZE, to);

    /* starting receive of array A in B */
    MPI_Recv(B, MSIZE, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
    printf("Proc %d receives %d integers from proc %d\n", 
        rank, MSIZE, from);

    /* print first content of arrays A and B */
    printf("Proc %d has A[0] = %d, B[0] = %d\n\n", rank, A[0], B[0]);

    /* Quit */
    MPI_Finalize();

    return 0;
}
