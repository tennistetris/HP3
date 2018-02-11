__global__ void

vectorSwap(float *A,float *B,const int size)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int l = sqrt((float)size);
    if (i < size)
    {
        int j = i/l;
        int k = i%l;
    	float temp;
        if((k%2)==0 && k!=l-1){
        	temp = A[i];
        	A[i] = A[i+1];
        	A[i+1] = temp;
        }

         __syncthreads();
    
        if(j<k){
       	    B[i] = A[k*l+j];
        }
        else{
            B[i] = A[i];
        }
    }

}