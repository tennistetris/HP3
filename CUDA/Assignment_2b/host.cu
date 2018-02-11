#include "headers.h"


void initializeMatrix(float * A,int numRows,int numCols){

    srand(100);
	for(int i = 0;i<numRows;i++){
		for(int j = 0;j<numCols;j++){
			A[i*numCols + j] = (float)rand()/(RAND_MAX/10.00);
		}
	}

}

void initializeArray(float *A,int numElements){
    srand(100);
    for(int i = 0;i<numElements;i++){
        A[i] = (float)rand()/(RAND_MAX/10.00);
    }
}

void displayMatrix(float * A,int numRows,int numCols){
	for(int i = 0;i<numRows;i++){
		for(int j = 0;j<numCols;j++){
			printf("%f ",A[i*numCols + j]);
		}
		printf("\n");
	}
}

void displayArray(float *A,int numElements){
    for(int i = 0;i<numElements;i++)printf("%f ",A[i]);
    printf("\n");
}

void naiveImplementation1D(float *A,float *Correct,int numElements){
    // Naive non-parallel implementation of the code for 1D Array.
    float pos1,pos2,pos3,pos4;
    for(int i = 0;i<numElements;i++){
        pos1=0;pos2=0;pos3=0;pos4=0;
        if(i-2>=0)pos1 = A[i-2];
        if(i-1>=0)pos2 = A[i-1];
        if(i+1<numElements)pos3 = A[i+1];
        if(i+2<numElements)pos4 = A[i+2];
        Correct[i] = (pos1 + pos2 + pos3 + pos4)/4;
    }
}

void naiveImplementation2D(float *A,float *Correct,int numRows,int numCols){
    // Naive non-parallel implementaion of the code for matrices
    float pos1 = 0,pos2 = 0,pos3 = 0,pos4 = 0,pos5 = 0,pos6 = 0,pos7 = 0,pos8 = 0;

    for(int i = 0;i<numRows;i++){
        for(int j = 0;j<numCols;j++){
            pos1 = 0;pos2 = 0;pos3 = 0;pos4 = 0;pos5 = 0;pos6 = 0;pos7 = 0;pos8 = 0;

            if((i-1)>=0 && (j-1)>=0)
                pos1 = A[(i-1)*numCols+(j-1)];
            if((i-1)>=0)
                pos2 = A[(i-1)*numCols+(j)];
            if((i-1)>=0 && (j+1)<numCols)
                pos3 = A[(i-1)*numCols+(j+1)];
            if((j-1)>=0)
                pos4 = A[i*numCols + (j-1)];
            if((j+1)<numCols)
                pos5 = A[i*numCols + (j+1)];
            if((i+1)<numRows && (j-1)>0)
                pos6 = A[(i+1)*numCols+(j-1)];
            if((i+1)<numRows)
                pos7 = A[(i+1)*numCols+(j)];
            if((i+1)<numRows && (j+1)<numCols)
                pos8 = A[(i+1)*numCols+(j+1)];
            
            Correct[i*numCols + j] = (pos1 + pos2 + pos3 +pos4 + pos5 + pos6 + pos7 + pos8)/8;
        } 
    }

    
}



void check2DImplementation(const float *A,const float *B,int numRows,int numCols){
    //checking the non parallel and parallel versions for the 2D implementation
    for(int i = 0;i<numRows;i++){
        for(int j = 0;j<numCols;j++){
            if(fabs(A[i*numCols + j]-B[i*numCols + j])>1e-5){
                printf("Error at index [%d][%d] .\nActual Value should be : %f .\nYour Value is : %f . \n",i,j,B[i*numRows+j],A[i*numRows+j]);
                return;
            }
        }
    }
    printf("PASS !!!\nCUDA 2D implementation matches Naive implementation.\n");
}


void check1DImplementation(const float *A,const float *B,int numElements){
    //checking the non parallel and parallel versions for the 1D implementation
    for(int i = 0;i<numElements;i++){
        if(fabs(A[i]-B[i])>1e-5){
            printf("Error at index [%d] .\nActual Value should be : %f .\nYour Value is : %f . \n",i,B[i],A[i]);
            return;
        }
    }
    printf("PASS !!!\nCUDA 1D implementation matches Naive implementation.\n");
}

void Implementaion1D(){
    // Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    printf("\n\n\n------PART 1--------\n\n\n\n1D\n");
    int numElements = 10;

	printf("%d elements\n", numElements);
	size_t size = numElements*sizeof(float);

	// Allocate the host input arrays A and B
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *Correct = (float *)malloc(size);
    //float convMask[5] = [.25,.25,0,.25,.25];

	//Initialize the array
	initializeArray(h_A,numElements);

	//Printing the array
	printf("Initial Array : \n");
    displayArray(h_A,numElements);
    
	// Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors B  in host memory to the device
    
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    // Launch kernel

    int blocksPerGrid,threadsPerBlock;
    blocksPerGrid = ceil((float)numElements/32);
    threadsPerBlock = 32;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    convolution1D<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Edited Array
    printf("Convoluted Array : \n"); 
    displayArray(h_B,numElements);

    naiveImplementation1D(h_A,Correct,numElements); 
    //printf("Correct : \n");
    //displayArray(Correct,numElements);
    check1DImplementation(h_B,Correct,numElements);

	// Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	
	// Free host memory
    free(h_A);
    free(h_B);
    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Implementation2D(){
    
    // Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    printf("\n\n\n------PART 2--------\n\n\n \n2D\n");
    int numRows = 10;
    int numCols = 10;

	printf("%dX%d elements\n", numRows,numCols);
	size_t size = (numRows*numCols)*sizeof(float);

	// Allocate the host input arrays A and B
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *Correct = (float *)malloc(size);
    //float convMask[5] = [.25,.25,0,.25,.25];

	//Initialize the matrix
    initializeMatrix(h_A,numRows,numCols);
    initializeMatrix(h_B,numRows,numCols);
    
    //Printing the matrix
	printf("Initial Matrix : \n");
    displayMatrix(h_A,numRows,numCols);
    
	// Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors B  in host memory to the device
    
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    // Launch kernel

    int gridDimX,gridDimY;
    gridDimX = ceil((float)numRows/8);
    //printf("gridDimX : %d \n",gridDimX);
    gridDimY = ceil((float)numCols/8);
    //printf("gridDimY : %d \n",gridDimY);
    dim3 blocksPerGrid(gridDimX,gridDimY,1);
    dim3 threadsPerBlock(8,8,1);
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,numRows,numCols);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Edited Array
    printf("Convoluted Matrix : \n"); 
    displayMatrix(h_B,numRows,numCols);

    naiveImplementation2D(h_A,Correct,numRows,numCols); 
    //printf("Correct : \n");
    //displayMatrix(Correct,numRows,numCols);
    check2DImplementation(h_B,Correct,numRows,numCols);

	// Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	
	// Free host memory
    free(h_A);
    free(h_B);
    
    // Reset the device and exit
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}
int main(){
	
    Implementaion1D();
    Implementation2D();

    return 0;
}
