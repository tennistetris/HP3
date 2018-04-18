#include <stdio.h>
#include <omp.h>
#include <math.h>


void Print_matrix(const int Matrix[100][100],int num){
	int i,j;
	for(i = 0;i<num;i++){
		for(j = 0;j<num;j++){
			printf("%d  ",Matrix[i][j]);
		}
		printf("\n");
	}
}

void slow_implementation(const int A[100][100],const int B[100][100],int C[100][100],int num){
	int sum;
	for(int i = 0;i<num;i++){
		for(int j = 0;j<num ;j++){
			int sum = 0;
			for(int k = 0 ;k<num;k++){
				sum += A[i][k]*B[k][j];
			}
			C[i][j] = sum;
		}
	}
}

void Check(int C_correct[100][100],int C[100][100],int num){

	for (int i = 0;i<num;i++){
		for(int j = 0;j<num;j++){
			if(fabs(C_correct[i][j] - C[i][j])>1e-5){
				printf("Wrong output at position [%d][%d] \n Correct Value = %d \n Your Value = %d \n",i,j,C_correct[i][j],C[i][j]);
				break;
			}
		}
	}
}


int main(){

	int A[100][100],B[100][100],C[100][100],C_correct[100][100];
	int num = 50,sum;
	int i,j;
	for (i = 0;i<num;i++){
		for(j = 0;j<num;j++){
			A[i][j] = i+j;
			B[i][j] = i+j;
			C[i][j] = 0;
		}
	}
	
	int chunk;
	double wtime;
	int thread_num = omp_get_max_threads();
	omp_set_num_threads(thread_num);
	chunk = 2*(num/thread_num) + 1;

	slow_implementation(A,B,C_correct,num);

	//PART 1

	printf(".....PART 1....\n");
	wtime = omp_get_wtime();

	#pragma omp parallel for 
	for(int i = 0;i<num;i++){
		for(int j = 0;j<num;j++){
			#pragma omp critical
			{
				int s = 0;
				for(int k = 0;k<num;k++){
					s+=A[i][k]*B[k][j];
				}
				C[i][j] = s;
			}	 
		}
	}

	wtime = omp_get_wtime() - wtime;
	printf("Time required using critical clause : %lf \n",wtime);
	Check(C_correct,C,num);
	
	wtime = omp_get_wtime();
	#pragma omp parallel for reduction (+:sum)
	for(int i = 0;i<num;i++){
		for(int j = 0;j<num;j++){
			int sum = 0;
			for(int k = 0;k<num;k++){
				sum += A[i][k]*B[k][j];
			}
			C[i][j] = sum;
		}
	}

	wtime = omp_get_wtime() - wtime;
	printf("Time required for the with reduction clause : %lf \n",wtime);
	Check(C_correct,C,num);	

	printf("......PART 2.....\n");
	wtime = omp_get_wtime();


	#pragma omp parallel for schedule(dynamic,chunk) private(sum)
	for(int i = 0;i<num;i++){
		for(int j = 0;j<num;j++){
				sum = 0;
				for(int k = 0;k<num;k++){
					sum+=A[i][k]*B[k][j];
				}	 
				C[i][j] = sum;
		}
	}

	wtime = omp_get_wtime() - wtime;
	printf("Time required for the  without collapse clause : %lf \n",wtime);
	Check(C_correct,C,num);

	//chunk = ceil(sqrt(chunk));
	wtime = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic,chunk)collapse(2) private(sum)
		for(int i = 0;i<num;i++){
			for(int j = 0;j<num;j++){
					sum = 0;
					for(int k = 0;k<num;k++){
						sum += A[i][k]*B[k][j];
					}
					C[i][j] = sum;
			}
		}
	wtime = omp_get_wtime() - wtime;
	printf("Time required for the with collapse clause : %lf \n",wtime);
	Check(C_correct,C,num);	


	return 0;
}
