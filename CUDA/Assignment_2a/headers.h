#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void vectorSwap(float *,float *,const int);