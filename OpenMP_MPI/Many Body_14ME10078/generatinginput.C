#include <stdio.h>
#include <stdlib.h>

const int NUM_SPHERES = 1000;
const int XDIM = 100;
const int YDIM = 200;
const int ZDIM = 400;

int main(){
    srand(42);
    for(int i = 0;i<NUM_SPHERES;i++){
        printf("%.7lf ",(float)rand()/RAND_MAX*XDIM);
        printf("%.7lf ",(float)rand()/RAND_MAX*YDIM);
        printf("%.7lf ",(float)rand()/RAND_MAX*ZDIM);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("%.14lf ",(double)rand()/RAND_MAX);
        printf("\n");
    }
}