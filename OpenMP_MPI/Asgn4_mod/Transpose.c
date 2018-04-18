/*
  The function used to compute the transpose of the matrix is `MPI_Alltoall`.
*/

#include <stdio.h>
#include "mpi.h"

#define Num_Proc 4
#define Matrix_data 10000
#define Sub_Matrix_data Matrix_data/Num_Proc

void sub_matrix_transpose(float *a, int n)
{
  int i=0, j;
  int i_t, j_t, k;
  float tmp;
  i_t = 0;
  k = -1;

  while(i<n)
  {
    k += n+1;
    j_t = k;
    i_t += i+1;
    for(j=i+1; j<n; j++)
    {
      tmp = a[i_t];
      a[i_t] = a[j_t];
      a[j_t] = tmp;
      j_t += n;
      i_t++;
      
    }
    i++;
  }
}

int main (int argc, char *argv[])
{
  float input[Matrix_data][Matrix_data]; // matrix before transpose
  float output[Matrix_data][Matrix_data]; // matrix after transpose
  int i, j;
  int nproc, myid;
  

  float temp_input[Matrix_data][Sub_Matrix_data];
  float temp_output[Matrix_data][Sub_Matrix_data];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if(nproc != Num_Proc)
  {
    if(myid == 0)
    {
      printf ("Number of processors input as %d,and my id is %d\n", nproc, Num_Proc);
    }
    MPI_Finalize();
    return 1;
  }

  
  if(myid == 0)
  {
    for(i=0; i<Matrix_data; i++)
    {
      for(j=0; j<Matrix_data; j++)
      {
        input[i][j] = (i*j/10000.0);
      }
    }
  }

  
  MPI_Bcast(&input[0][0], Matrix_data*Matrix_data, MPI_FLOAT, 0, MPI_COMM_WORLD);

  
  for(i=0; i<Matrix_data; i++)
  {
    for(j=0; j<Sub_Matrix_data; j++)
    {
      temp_input[i][j] = input[i][Sub_Matrix_data*myid + j];
    }
  }

  MPI_Alltoall(&temp_input[0][0], Sub_Matrix_data*Sub_Matrix_data, MPI_FLOAT, &temp_output[0][0], Sub_Matrix_data*Sub_Matrix_data, MPI_FLOAT, MPI_COMM_WORLD);

  
  for(i=0; i<Num_Proc; i++)
  {
    sub_matrix_transpose(&temp_output[i*Sub_Matrix_data][0], Sub_Matrix_data);
  }

  
  MPI_Gather(&temp_output[0][0], Matrix_data*Sub_Matrix_data, MPI_FLOAT, &output[0][0], Matrix_data*Sub_Matrix_data, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Finalize ();
  return 0;
}