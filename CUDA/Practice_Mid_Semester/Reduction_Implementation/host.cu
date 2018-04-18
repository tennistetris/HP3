#include "headers.h"

void initializeArray(float *h_A,int n){
    for(int i = 0;i<n;i++){
        //h_A[i] = 1;
        h_A[i] = -1 + 2*(rand()/(float)RAND_MAX);
    }
}

float naive_sum(float *h_A,int n){
    float sum = 0;
    for(int i = 0;i<n;i++){
        sum +=h_A[i];
    }
    return sum;
}

int main(){
    printf("Enter the number of elements  : ");
    int numElements;
    scanf("%d",&numElements);
    printf("You entered %d : \n",numElements);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = numElements*sizeof(float);
    
    // Allocate the host input array A
    float *h_A = (float *)malloc(size);
    
    //Initialize the array 
    initializeArray(h_A,numElements);
    float correct_ans = naive_sum(h_A,numElements);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    int blocksPerGrid,threadsPerBlock;
    blocksPerGrid = 1;
    threadsPerBlock = numElements;
    reduction_sum<<<blocksPerGrid,threadsPerBlock>>>(d_A,numElements);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    float ans = 0;
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_A,d_A,size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ans = h_A[0];

    //printf("Modified Array : \n");
    //for(int i = 0;i<numElements;i++)printf(" %f ",h_A[i]);

    
    if(fabs(correct_ans - ans)>1e-5){
        printf("\nERROR : WRONG IMPLEMENTATION \nActual value should be : %f \n Your Answer : %f\n",correct_ans,ans);
    }
    else{
        printf("\nAns : %f\n",ans);
    }

    return 0;
}