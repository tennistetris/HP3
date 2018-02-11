#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void convolution1D(float *,float *,const int);
__global__ void convolution2D(float *,float *,const int,const int);