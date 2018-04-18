#include<stdio.h>
#include<unistd.h>
#include<mpi.h>

int main(int argc,char *argv[])
{
	int size,id,N=30000,i,x,y;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&id);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Barrier(MPI_COMM_WORLD);
	int lhit = 0;
	srand((unsigned)(time(0)));
	int lN= N/size;
	for(i = 0; i<lN;i++)
	{
		x = ((double)rand())/((double)RAND_MAX);
		y = ((double)rand())/((double)RAND_MAX);
		if (((x*x) + (y*y)) <= 1) lhit++;
	}
	
	int hit=0;
	MPI_Allreduce(&lhit,&hit,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	double PI;
	PI=(hit*4)/((double)N);
	MPI_Barrier(MPI_COMM_WORLD);
	if (id == 0) {
	printf("Estimate of Pi:%24.16f\n",est);
}
MPI_Finalize();
	return 0;
}
