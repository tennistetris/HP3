__global__ void 
reduction_sum(float *A,int num_elements){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<num_elements){

        for(int stride = 1;stride<num_elements;stride*=2){
        
            __syncthreads();
            if(i%(2*stride) == 0){
                float temp = 0;
                if(i+stride<num_elements)temp = A[i + stride];
                float partial_sum = A[i] + temp;
                A[i] = partial_sum;
            }
        
        }

    }
}