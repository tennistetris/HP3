/* Exercise: Ring
 *
 * In this exercise you will communicate an integer
 * among process in a circular topology, so that
 * every process has a left and a right neighbor
 * to receive and to send data respectivelly.
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Status status;
    int rank, size, tag, next, from;

    int num = 1;
    
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* print banner */
    if (rank == 0) {
        printf("\nCircular ring example: when a process receives a msg\n");
        printf("from its left neighbor, it passes to its right neighbor.\n");
        printf("Iteration counter is decremented each time the msg pass\n");
        printf("through master process.\n\n");

        fflush(stdout);
    }

    /* Calculate the rank of the next process in the ring.  
     * Use the modulus operator so that the last process 
     * "wraps around" to rank zero. */
    
    tag = 201;
    next = (rank + 1) % size;
    from = (rank + size - 1) % size;
    
    /* If we are the "console" process, get a integer from the user
       to specify how many times we want to go around the ring */
    if (rank == 0) {
        printf("Enter the number of times around the ring: \n");

        fflush(stdout);

        scanf("%d", &num);

        if (num <= 0) /* at least one turn */
            num = 1;
    
        printf("Proc %d sends a bag with %d sandwich to proc %d\n", rank, num, next);

        fflush(stdout);

        /* main process start the ring sending num to the next process */
        MPI_Send(&num, 1, MPI_INT, next, tag, MPI_COMM_WORLD); 
    }
    
    /* Pass the message around the ring.  The exit mechanism works as
       follows: the message (a positive integer) is passed around the
       ring.  Each time it passes rank 0, it is decremented.  When each
       processes receives the 0 message, it passes it on to the next
       process and then quits.  By passing the 0 first, every process
       gets the 0 message and can quit normally. */
    
    do {
        /* each process waits for an incoming message with a receive */ 
        MPI_Recv(&num, 1, MPI_INT, from, tag, MPI_COMM_WORLD, &status);

        printf("Proc %d receives a bag with %d sandwich\n", rank, num);
    
    	fflush(stdout);
        /* if message passes through process 0, num is decremented */
        if (rank == 0) {

            num = num - 1;

            printf("\n* * * Process 0 eats one sandwich * * *\n\n");
        	fflush(stdout);
        }
    
        /* send the message to the next process */
        MPI_Send(&num, 1, MPI_INT, next, tag, MPI_COMM_WORLD);

        printf("Proc %d sends a bag with %d sandwich to proc %d\n", rank, num, next);

        fflush(stdout);

    } while (num > 0);



    /* END OF THE RING */
    printf("Process %d leaves the ring\n", rank);
    
    /* The last process sends 0 to process 0, which needs to
       be received before the program can exit */
    if (rank == 0) {
        MPI_Recv(&num, 1, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
        printf("\n... process 0 starves\n\n");
    }
    
    /* Quit */
    MPI_Finalize();

    return 0;
}
