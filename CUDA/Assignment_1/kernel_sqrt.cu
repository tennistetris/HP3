
__global__ void
process_kernel3(const float *A, float *C, const int size)
{
    int threads_per_block = blockDim.x*blockDim.y*blockDim.z;
    int i = blockIdx.z*(gridDim.x*gridDim.y)*threads_per_block + blockIdx.y*(gridDim.x)*(threads_per_block) + blockIdx.x*threads_per_block //Specifying the thread no. 
            +threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*(blockDim.x)+threadIdx.x;

    if (i < size)
    {
        C[i] = sqrt(A[i]);
    }
}



