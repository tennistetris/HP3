__global__ void
naiveKernel(float *A,float *A_out,const int n){
    //First make a function that works for the input size 2*2
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    sdata[tid] = 0;
    if(i<n)
        sdata[tid] = A[i];
    
    for(unsigned int s=4;s<blockDim.x;s*=2){
        if(tid%(2*s)==0)
            if((tid+s)<blockDim.x){
                sdata[tid] += sdata[tid+s];
                sdata[tid+1] += sdata[tid+1+s];
                sdata[tid+2] += sdata[tid+2+s];
                sdata[tid+3] += sdata[tid+3+s];
            }
        __syncthreads();
    }
    if(tid<4)
        A_out[4*blockIdx.x+tid] = sdata[tid];
}

__global__ void
optimKernel1(float *A,float *A_out,const int n){
    //Getting rid of the divergent threads
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    sdata[tid] = 0;
    if(i<n)
        sdata[tid] = A[i];
    
    for(unsigned int s=4;s<blockDim.x;s*=2){
        int index = 2*s*(tid/4);    
        if(index< blockDim.x)
            sdata[index + tid%4] += sdata[index+ s + tid%4];
        __syncthreads();
    }
    if(tid<4)
        A_out[4*blockIdx.x+tid] = sdata[tid];
}

__global__ void
optimKernel2(float *A,float *A_out,const int n){
    //Sequential Addressing
    //NOTE : This thing works since blockDim is 1024
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    sdata[tid] = 0;
    if(i<n)
        sdata[tid] = A[i];
    
    for(unsigned int s=blockDim.x/2;s>=4;s/=2){
        if(tid<s){
            sdata[tid] += sdata[tid+s]; 
        }
        __syncthreads();
    }
    if(tid<4)
        A_out[4*blockIdx.x+tid] = sdata[tid];
}

__global__ void
optimKernel3(float *A,float *A_out,const int n){
    //Idle Threads
    //NOTE : This thing works since blockDim is 1024
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockDim.x*(blockIdx.x*2)+threadIdx.x;

    sdata[tid] = 0;
    if(i<n)
        sdata[tid] = A[i] + (1 - (i+blockDim.x)/n)*A[i + blockDim.x];
    
    for(unsigned int s=blockDim.x/2;s>2;s/=2){
        if(tid<s){
            sdata[tid] += sdata[tid+s]; 
        }
        __syncthreads();
    }
    if(tid<4)
        A_out[4*blockIdx.x+tid] = sdata[tid];
}

__global__ void
optimKernel4(float *A,float *A_out,const int n){
    //Unwrapping the last roll
    //NOTE : This thing works since blockDim is 1024
    //NOTE :- Code incorrect,doesn't work , see to it
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockDim.x*(blockIdx.x*2)+threadIdx.x;

    sdata[tid] = 0;
    if(i<n)
        sdata[tid] = A[i] + (1 - (i+blockDim.x)/n)*A[i + blockDim.x];
    
    for(unsigned int s=blockDim.x/2;s>32;s/=2){
        if(tid<s){
            sdata[tid] += sdata[tid+s]; 
        }
        __syncthreads();
    }
    
    __syncthreads();

    if(tid<32){
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
    }
    
    if(tid<4)
        A_out[4*blockDim.x + tid] = sdata[tid];

}


