__global__ void

convolution1D(float *A,float *B,const int size)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    

    if (i < size)
    { 
        float pos1=0,pos2=0,pos3=0,pos4=0;  
        if(i>1)pos1 = A[i-2];
        if(i>0)pos2 = A[i-1];
        if(i<size-1)pos3 = A[i+1];
        if(i<size-2) pos4 = A[i+2];
        B[i] = (pos1 + pos2 + pos3 +pos4)/4; 
    }

}