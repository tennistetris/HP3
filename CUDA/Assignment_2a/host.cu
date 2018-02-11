#include "headers.h"


void initializeMatrix(float * A,int numElements){

    srand(100);
	for(int i = 0;i<numElements;i++){
		for(int j = 0;j<numElements;j++){
			A[i*numElements + j] = (float)rand()/(RAND_MAX/10.00);
		}
	}

}

void displayMatrix(float * A,int numElements){
	for(int i = 0;i<numElements;i++){
		for(int j = 0;j<numElements;j++){
			printf("%f ",A[i*numElements + j]);
		}
		printf("\n");
	}
}
void naiveImplementation(float *A,float *Correct,int size){
    // Naive non-parallel implementaion of the code
    int len = sqrt(size),j,k;
    for(int i = 0;i<size;i++){
        j = i/len;
        k = i%len;
        if((k % 2) == 0 && (k != len-1)){
            Correct[i] = A[i+1];
            Correct[i+1] = A[i];
        }
        
    }
    for(int i = 0;i<size;i++){
        j = i/len;
        k = i%len;
        
        if(j<k)Correct[i] = Correct[k*len + j];
    }
    //Corner Case for odd sized matrices
    if(len%2)Correct[size - 1] = A[size - 1];

}

void check(const float *A,const float *B,int size){
    //checking the non parallel and parallel versions
    int len = sqrt(size);
    for(int i = 0;i<size;i++){
        if(fabs(A[i]-B[i]>1e-5)){
            printf("Error at index [%d][%d] .\nActual Value should be : %f .\nYour Value is : %f . \n",i/len,i%len,B[i],A[i]);
            return;
        }
    }
    printf("PASS !!!\nCUDA implementation matches Naive implementation.\n");
}

int main(){
	
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    printf("--PART 1-- \n");
    int numElements = 10;

	printf("%d elements\n", numElements);
	size_t size = numElements*sizeof(float);

	// Allocate the host input matrix A
	float *h_A = (float *)malloc(size*size);
    float *h_B = (float *)malloc(size*size);
	float *h_Correct = (float *)malloc(size*size);


	//Initialize the matrix
	initializeMatrix(h_A,numElements);
	//initializeMatrix(h_B,numElements);

	//Printing the matrix
	printf("Initial Matrix : \n");
	displayMatrix(h_A,numElements);
	// Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size*size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A  in host memory to the device
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size*size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size*size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	// Copy the host input vectors B  in host memory to the device
    
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_B, h_B, size*size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    // Launch kernel

    int blocksPerGrid,threadsPerBlock;
    blocksPerGrid = numElements;
    threadsPerBlock = numElements;
    int arraySize = numElements*numElements;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorSwap<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,arraySize);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size*size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Edited Matrix
    printf("Edited Matrix \n"); 
    displayMatrix(h_B,numElements);

    naiveImplementation(h_A,h_Correct,numElements*numElements); 
    //printf("Correct : \n");
    //displayMatrix(h_Correct,numElements);
    check(h_B,h_Correct,numElements*numElements);

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

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	return 0;
}
