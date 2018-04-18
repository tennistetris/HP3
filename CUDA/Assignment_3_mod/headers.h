#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

__global__ void naiveKernel(float *,float *,const int);
__global__ void optimKernel1(float *,float *,const int);
__global__ void optimKernel2(float *,float *,const int);
__global__ void optimKernel3(float *,float *,const int);
__global__ void optimKernel4(float *,float *,const int);