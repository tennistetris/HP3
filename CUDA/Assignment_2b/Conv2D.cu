__global__ void

convolution2D(float *A,float *B,const int numRows,const int numCols)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;

    if (i<numRows && j<numRows)
    { 
        float pos1=0,pos2=0,pos3=0,pos4=0,pos5=0,pos6=0,pos7=0,pos8=0;  
        if((i-1)>=0 && (j-1)>=0)pos1 = A[(i-1)*numCols+(j-1)];
        if((i-1)>=0)pos2 = A[(i-1)*numCols+(j)];
        if((i-1)>=0 && (j+1)<numCols)pos3 = A[(i-1)*numCols+(j+1)];
        if((j-1)>=0)pos4 = A[(i)*numCols+(j-1)];
        if((j+1)<numCols)pos5 = A[(i)*numCols+(j+1)];
        if((i+1)<numRows && (j-1)>0)pos6 = A[(i+1)*numCols+(j-1)];
        if((i+1)<numRows)pos7 = A[(i+1)*numCols+(j)];
        if((i+1)<numRows && (j+1)<numCols)pos8 = A[(i+1)*numCols+(j+1)];

        B[i*numCols + j] = (pos1 + pos2 + pos3 +pos4 + pos5 + pos6 + pos7 + pos8)/8; 
    }

}