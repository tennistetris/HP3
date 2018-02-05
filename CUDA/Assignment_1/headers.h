#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kernel1(const float*, const float*,float*,const int);
__global__ void process_kernel2(const float*,float*,const int);
__global__ void process_kernel3(const float*,float*,const int);

