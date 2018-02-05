# include <stdio.h>
# include <omp.h>

int main ( int argc, char *argv[] ) {

    int id;
    double wtime;
    
    printf ("Number of processors available = %d\n", omp_get_num_procs());
    printf ("Number of threads =%d\n", omp_get_max_threads());
    wtime = omp_get_wtime();
    printf("OUTSIDE the parallel region.\n");
    id = omp_get_thread_num ( );
    printf ( "HELLO from process %d\n Going INSIDE the parallel region:\n ", id ) ;
    
    # pragma omp parallel private(id)
    {
        id = omp_get_thread_num ( );
        printf (" Hello from process %d\n", id );   
    }
    
    wtime = omp_get_wtime() - wtime;
    printf ( "Back OUTSIDE the parallel region.\nNormal end of execution.\nElapsed wall clock time = %f\n", wtime );
    return 0;
}