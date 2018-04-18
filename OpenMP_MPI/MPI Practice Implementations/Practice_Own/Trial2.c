#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc,char * argv[]){
    int ret,nprocs,myid;

    ret = MPI_Init(&argc,&argv);
    ret = MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    ret = MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    printf("Number of processors %d .. my id is %d\n",nprocs,myid);

    ret = MPI_Finalize();

    return 0;

}