#include<stdio.h>
#include<omp.h>

int main(){
    #ifdef _OPENMP
        printf("Compiled with OpenMP support:%d \n",_OPENMP);
    #else
        printf("Compiled for serial execution.");
    #endif

    #pragma omp parallel
    {
        printf("Hello world! \n");
    }
    return 0;
}