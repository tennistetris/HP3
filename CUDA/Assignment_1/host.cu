#include "headers.h"
#include <math.h>

 void Part_1(){
     // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    printf("--PART 1-- \n");
    int numElements = 10000;
    
    //printf("Enter the number of elements( < 16384) : ");
    //scanf("%d",&numElements);

    printf("%d elements\n", numElements);
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch process_kernel1
    dim3 threadsPerBlock(4,2,2);
    dim3 blocksPerGrid(32,32,1);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    process_kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sin(h_A[i]) + cos(h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

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
    
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

 }

 void Part_2(){
    // Error code to check return values for CUDA calls
   cudaError_t err = cudaSuccess;

   // Print the vector length to be used, and compute its size
   printf("-------------PART 2--------------------- \n");
   int numElements = 10000;
   
   //printf("Enter the number of elements( < 16384) : ");
   //scanf("%d",&numElements);
   printf("%d elements\n", numElements);
   size_t size = numElements * sizeof(float);
   // Allocate the host input vector A
   float *h_A = (float *)malloc(size);

   // Allocate the host output vector C
   float *h_C = (float *)malloc(size);

   // Verify that allocations succeeded
   if (h_A == NULL || h_C == NULL)
   {
       fprintf(stderr, "Failed to allocate host vectors!\n");
       exit(EXIT_FAILURE);
   }

   // Initialize the host input vectors
   for (int i = 0; i < numElements; ++i)
   {
       h_A[i] = rand()/(float)RAND_MAX;
   }

   // Allocate the device input vector A
   float *d_A = NULL;
   err = cudaMalloc((void **)&d_A, size);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Allocate the device output vector C
   float *d_C = NULL;
   err = cudaMalloc((void **)&d_C, size);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Copy the host input vector A in host memory to the device input vector in
   // device memory
   printf("Copy input data from the host memory to the CUDA device\n");
   err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Launch process_kernel2
   dim3 threadsPerBlock(8,8,8);
   int block_size = 8*8*8;
   int blocks_y = 1 + ceil(numElements/((2*1)*block_size));
   //printf("Blocks_y : %d",blocks_y);
   dim3 blocksPerGrid(2,blocks_y,1);
   
   //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
   process_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_C, numElements);
   err = cudaGetLastError();

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to launch process_kernel2 (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Copy the device result vector in device memory to the host result vector
   // in host memory.
   printf("Copy output data from the CUDA device to the host memory\n");
   err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Verify that the result vector is correct
   for (int i = 0; i < numElements; ++i)
   {
       if (fabs(log(h_A[i]) - h_C[i]) > 1e-5)
       {
           fprintf(stderr, "Result verification failed at element %d!\n", i);
           exit(EXIT_FAILURE);
       }
   }

   printf("Test PASSED\n");

   // Free device global memory
   err = cudaFree(d_A);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   
   err = cudaFree(d_C);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   

   // Free host memory
   free(h_A);
   free(h_C);

}


void Part_3(){
    // Error code to check return values for CUDA calls
   cudaError_t err = cudaSuccess;

   // Print the vector length to be used, and compute its size
   printf("-------------PART 3--------------------- \n");
   int numElements = 10000;
   
   //printf("Enter the number of elements( < 16384) : ");
   //scanf("%d",&numElements);
   printf("%d elements\n", numElements);
   size_t size = numElements * sizeof(float);

   // Allocate the host input vector A
   float *h_A = (float *)malloc(size);

   // Allocate the host output vector C
   float *h_C = (float *)malloc(size);

   // Verify that allocations succeeded
   if (h_A == NULL || h_C == NULL)
   {
       fprintf(stderr, "Failed to allocate host vectors!\n");
       exit(EXIT_FAILURE);
   }

   // Initialize the host input vectors
   for (int i = 0; i < numElements; ++i)
   {
       h_A[i] = rand()/(float)RAND_MAX;
   }

   // Allocate the device input vector A
   float *d_A = NULL;
   err = cudaMalloc((void **)&d_A, size);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Allocate the device output vector C
   float *d_C = NULL;
   err = cudaMalloc((void **)&d_C, size);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Copy the host input vector A in host memory to the device input vector in
   // device memory
   printf("Copy input data from the host memory to the CUDA device\n");
   err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Launch process_kernel3
   
   dim3 threadsPerBlock(128,4,1);
   int block_size = 128*4;
   int blocks_x = 1+ ceil(numElements/((1*1)*block_size));
   //printf("blocks_x : %d ",blocks_x);
   dim3 blocksPerGrid(blocks_x,1,1);
   //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
   process_kernel3<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_C, numElements);
   err = cudaGetLastError();

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to launch process_kernel3 (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Copy the device result vector in device memory to the host result vector
   // in host memory.
   printf("Copy output data from the CUDA device to the host memory\n");
   err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   // Verify that the result vector is correct
   for (int i = 0; i < numElements; ++i)
   {
       if (fabs(sqrt(h_A[i]) - h_C[i]) > 1e-5)
       {
           fprintf(stderr, "Result verification failed at element %d!\n", i);
           exit(EXIT_FAILURE);
       }
   }

   printf("Test PASSED\n");

   // Free device global memory
   err = cudaFree(d_A);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   
   err = cudaFree(d_C);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   

   // Free host memory
   free(h_A);
   free(h_C);

}

 /**
 * Host main routine
 */

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    Part_1();
    Part_2();
    Part_3();
    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}


