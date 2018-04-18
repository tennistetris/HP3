/*
Used MPI_Scatter and MPI_Reduce
*/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int SIZE=1e4;    //Size of the input matrix
int RANGE=250;   //Actual range is 250((max(i)+max(j))/80) extended to 300

int main(int argc,char * argv[]){
    int *Arr;
    Arr = (int*)malloc(SIZE*SIZE*sizeof(int));
    int hist[RANGE];
    int temp_hist[RANGE];

    int nprocs,myid;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if(SIZE%nprocs!=0)
    {
        if(myid == 0)
        {
            printf ("Number of processors %d should divide Array size %d\n", nprocs,SIZE);
        }
        MPI_Finalize();
        return 1;
    }
    int size_per_proc = SIZE/nprocs;

    if(myid == 0){

        for(int i=0; i<SIZE; i++)
            for(int j=0; j<SIZE; j++)
                Arr[i*SIZE + j] = ((i+j)/80);

        for(int i = 0;i<RANGE;i++)
            hist[i] = 0;
    }

    for(int i = 0;i<RANGE;i++)
        temp_hist[i] = 0;
    
    int *local_arr;
    int local_size = size_per_proc*SIZE;
    local_arr = (int*)malloc(local_size*sizeof(int));
    MPI_Scatter(Arr,size_per_proc*SIZE,MPI_INT,local_arr,size_per_proc*SIZE,MPI_INT,0,MPI_COMM_WORLD);

    for(int i = 0;i<local_size;i++)
        temp_hist[local_arr[i]]++;

    MPI_Reduce(temp_hist,hist,RANGE,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    if(myid == 0){
        printf("OUTPUT HISTOGRAM :- \n");
        for(int i = 0;i<RANGE;i++){
            printf("%d -- %d\n",i,hist[i]);
        }
    }

    MPI_Finalize();
    return 1;
} 