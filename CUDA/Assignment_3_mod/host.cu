#include "headers.h"

const int SIZE = 2;

void printDevProp(cudaDeviceProp* prop){
    printf(".................................................\n\n");
    printf("Device Name :- %s\n",prop->name);
    printf("Wrap size :- %d\n",prop->warpSize);
    printf("Multi Processor Count :- %d\n",prop->multiProcessorCount);
    printf("Max threads per block :- %d\n",prop->maxThreadsPerBlock);
    printf("Total   device memory :- %zu\n",prop->totalGlobalMem);
    printf("Shared memory per block :- %zu\n",prop->sharedMemPerBlock);
    printf("Max threads dim :- (%d,%d,%d)\n",prop->maxThreadsDim[0],prop->maxThreadsDim[1],prop->maxThreadsDim[2]);
    printf("Max grid size :- (%d,%d,%d)\n",prop->maxGridSize[0],prop->maxGridSize[1],prop->maxGridSize[2]);
    printf("\n.................................................\n\n");
        
}

double getWallTime()     //stackoverflow
{
    struct timeval time;
    if(gettimeofday(&time, NULL))
    {
        return 0;
    }
    double wallTime = (double)time.tv_sec + (double)time.tv_usec * 0.000001;
    return wallTime;
}

void fillArray(float * Arr,const int n){
    unsigned int num_max = n*(SIZE*SIZE);//no of elements = #matrices * #elements_per_matrix
    for(int i = 0;i<num_max;i++){
        if(i%10000 == 0)
            srand(i);
        Arr[i] = (rand()/(float)(RAND_MAX))/(float(n)/1000);
    }
}

void printArray(float * Arr,const int n){
    unsigned int num_max = n;
    for(int i = 0;i<num_max;i++){
        if(i%(SIZE*SIZE)==0)
            printf("\n");
        printf("%f  ",Arr[i]);
    }
    printf("\n");
    printf("\n%d elements\n",n);
}

void naiveImplementation(const int warp_size,const int procs,const int max_threads_per_block){
    int n;
    int response;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t arr_size;

    printf("Enter no of Arrays : ");
    scanf("%d",&n);

    //Defining the matrices
    float* h_A;
    arr_size = (SIZE*SIZE)*n*sizeof(float);
    h_A = (float*)malloc(arr_size);
    fillArray(h_A,n);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**)&d_A,arr_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Defining the blocksize and the grid size
    int threads_per_block = warp_size*(max_threads_per_block/warp_size);
    int num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
    
    //Defining h_sdata;
    float* h_sdata;
    size_t sdata_size = (SIZE*SIZE)*num_blocks*sizeof(float);
    h_sdata = (float*)malloc(sdata_size);

    // Allocate the device vector sdata 
    float *d_sdata = NULL;
    err = cudaMalloc((void**)&d_sdata,sdata_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printArray(h_A , arr_size/(sizeof(float)));
    
    //Main recursive kernel invocation
    while(1){

        //Kernel Invocation
        printf("Launched kernel with %d blocks and %d threads per block \n",num_blocks,threads_per_block);
        naiveKernel<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(d_A,d_sdata,n*(SIZE*SIZE));
        
        //Updation steps
        //.............

        //Copying back output array
        err = cudaMemcpy(h_sdata,d_sdata,sdata_size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy output array from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printArray(h_sdata,sdata_size/sizeof(float));

        //TERMINATION STATEMENT
        if(sdata_size == SIZE*SIZE*sizeof(float))
            break;  
        
        //Freeing unused memory
        free(h_A);
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Freeing unused memory
        err = cudaFree(d_sdata);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //updation of n and num_blocks 
        n = num_blocks;
        num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
        
        //Generating input for the next iter.
        arr_size = sdata_size;
        h_A = h_sdata;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A,arr_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vector A  in host memory to the device
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printArray(h_A , arr_size/(sizeof(float)));

        //Generating the output vector for the next iteration 
        free(h_sdata);
        sdata_size = num_blocks*(SIZE*SIZE)*sizeof(float);
        h_sdata = (float*)malloc(sdata_size);
        
        // Allocate the device output vector sdata
        err = cudaMalloc((void**)&d_sdata,sdata_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //Copying the new output vector to device
        printf("Coping output array from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_sdata, h_sdata,sdata_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array sdata from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }     

    }
    
    //float *result;
    /*{
    result = (float*)malloc(SIZE*SIZE*sizeof(float));
    size_t result_size = SIZE*SIZE*sizeof(float);
    printf("\n\nAnswer :- \n\n");
    //Copying back output array
    err = cudaMemcpy(result,d_sdata,result_size,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    }*/
    printf("Result :- \n");
    printArray(h_sdata , SIZE*SIZE);
    
}

void optimised1(const int warp_size,const int procs,const int max_threads_per_block){
    int n;
    int response;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t arr_size;

    printf("Enter no of Arrays : ");
    scanf("%d",&n);

    //Defining the matrices
    float* h_A;
    arr_size = (SIZE*SIZE)*n*sizeof(float);
    h_A = (float*)malloc(arr_size);
    fillArray(h_A,n);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**)&d_A,arr_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Defining the blocksize and the grid size
    int threads_per_block = warp_size*(max_threads_per_block/warp_size);
    int num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
    
    //Defining h_sdata;
    float* h_sdata;
    size_t sdata_size = (SIZE*SIZE)*num_blocks*sizeof(float);
    h_sdata = (float*)malloc(sdata_size);

    // Allocate the device vector sdata 
    float *d_sdata = NULL;
    err = cudaMalloc((void**)&d_sdata,sdata_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printArray(h_A , arr_size/(sizeof(float)));
    
    //Main recursive kernel invocation
    while(1){

        //Kernel Invocation
        printf("Launched kernel with %d blocks and %d threads per block \n",num_blocks,threads_per_block);
        optimKernel1<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(d_A,d_sdata,n*(SIZE*SIZE));
        
        //Updation steps
        //.............

        //Copying back output array
        err = cudaMemcpy(h_sdata,d_sdata,sdata_size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy output array from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printArray(h_sdata,sdata_size/sizeof(float));

        //TERMINATION STATEMENT
        if(sdata_size == SIZE*SIZE*sizeof(float))
            break;  
        
        //Freeing unused memory
        free(h_A);
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Freeing unused memory
        err = cudaFree(d_sdata);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //updation of n and num_blocks 
        n = num_blocks;
        num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
        
        //Generating input for the next iter.
        arr_size = sdata_size;
        h_A = h_sdata;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A,arr_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vector A  in host memory to the device
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printArray(h_A , arr_size/(sizeof(float)));

        //Generating the output vector for the next iteration 
        free(h_sdata);
        sdata_size = num_blocks*(SIZE*SIZE)*sizeof(float);
        h_sdata = (float*)malloc(sdata_size);
        
        // Allocate the device output vector sdata
        err = cudaMalloc((void**)&d_sdata,sdata_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //Copying the new output vector to device
        printf("Coping output array from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_sdata, h_sdata,sdata_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array sdata from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }     

    }
    
    //float *result;
    /*{
    result = (float*)malloc(SIZE*SIZE*sizeof(float));
    size_t result_size = SIZE*SIZE*sizeof(float);
    printf("\n\nAnswer :- \n\n");
    //Copying back output array
    err = cudaMemcpy(result,d_sdata,result_size,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    }*/
    printf("Result :- \n");
    printArray(h_sdata , SIZE*SIZE);
    
}


void optimised2(const int warp_size,const int procs,const int max_threads_per_block){
    int n;
    int response;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t arr_size;

    printf("Enter no of Arrays : ");
    scanf("%d",&n);

    //Defining the matrices
    float* h_A;
    arr_size = (SIZE*SIZE)*n*sizeof(float);
    h_A = (float*)malloc(arr_size);
    fillArray(h_A,n);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**)&d_A,arr_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Defining the blocksize and the grid size
    int threads_per_block = warp_size*(max_threads_per_block/warp_size);
    int num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
    
    //Defining h_sdata;
    float* h_sdata;
    size_t sdata_size = (SIZE*SIZE)*num_blocks*sizeof(float);
    h_sdata = (float*)malloc(sdata_size);

    // Allocate the device vector sdata 
    float *d_sdata = NULL;
    err = cudaMalloc((void**)&d_sdata,sdata_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printArray(h_A , arr_size/(sizeof(float)));
    
    //Main recursive kernel invocation
    while(1){

        //Kernel Invocation
        printf("Launched kernel with %d blocks and %d threads per block \n",num_blocks,threads_per_block);
        optimKernel2<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(d_A,d_sdata,n*(SIZE*SIZE));
        
        //Updation steps
        //.............

        //Copying back output array
        err = cudaMemcpy(h_sdata,d_sdata,sdata_size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy output array from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printArray(h_sdata,sdata_size/sizeof(float));

        //TERMINATION STATEMENT
        if(sdata_size == SIZE*SIZE*sizeof(float))
            break;  
        
        //Freeing unused memory
        free(h_A);
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Freeing unused memory
        err = cudaFree(d_sdata);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //updation of n and num_blocks 
        n = num_blocks;
        num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
        
        //Generating input for the next iter.
        arr_size = sdata_size;
        h_A = h_sdata;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A,arr_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vector A  in host memory to the device
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printArray(h_A , arr_size/(sizeof(float)));

        //Generating the output vector for the next iteration 
        free(h_sdata);
        sdata_size = num_blocks*(SIZE*SIZE)*sizeof(float);
        h_sdata = (float*)malloc(sdata_size);
        
        // Allocate the device output vector sdata
        err = cudaMalloc((void**)&d_sdata,sdata_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //Copying the new output vector to device
        printf("Coping output array from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_sdata, h_sdata,sdata_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array sdata from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }     

    }
    
    //float *result;
    /*{
    result = (float*)malloc(SIZE*SIZE*sizeof(float));
    size_t result_size = SIZE*SIZE*sizeof(float);
    printf("\n\nAnswer :- \n\n");
    //Copying back output array
    err = cudaMemcpy(result,d_sdata,result_size,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    }*/
    printf("Result :- \n");
    printArray(h_sdata , SIZE*SIZE);
    
}

void optimised3(const int warp_size,const int procs,const int max_threads_per_block){
    int n;
    int response;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t arr_size;

    printf("Enter no of Arrays : ");
    scanf("%d",&n);

    //Defining the matrices
    float* h_A;
    arr_size = (SIZE*SIZE)*n*sizeof(float);
    h_A = (float*)malloc(arr_size);
    fillArray(h_A,n);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**)&d_A,arr_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Defining the blocksize and the grid size
    int threads_per_block = warp_size*(max_threads_per_block/warp_size);
    int num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);

    num_blocks = ceil(num_blocks/2.0);
    
    //Defining h_sdata;
    float* h_sdata;
    size_t sdata_size = (SIZE*SIZE)*num_blocks*sizeof(float);
    h_sdata = (float*)malloc(sdata_size);

    // Allocate the device vector sdata 
    float *d_sdata = NULL;
    err = cudaMalloc((void**)&d_sdata,sdata_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printArray(h_A , arr_size/(sizeof(float)));
    
    //Main recursive kernel invocation
    while(1){

        //Kernel Invocation
        printf("Launched kernel with %d blocks and %d threads per block \n",num_blocks,threads_per_block);
        optimKernel3<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(d_A,d_sdata,n*(SIZE*SIZE));
        
        //Updation steps
        //.............

        //Copying back output array
        err = cudaMemcpy(h_sdata,d_sdata,sdata_size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy output array from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printArray(h_sdata,sdata_size/sizeof(float));

        //TERMINATION STATEMENT
        if(sdata_size == SIZE*SIZE*sizeof(float))
            break;  
        
        //Freeing unused memory
        free(h_A);
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Freeing unused memory
        err = cudaFree(d_sdata);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //updation of n and num_blocks 
        n = num_blocks;
        num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
        
        //Generating input for the next iter.
        arr_size = sdata_size;
        h_A = h_sdata;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A,arr_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vector A  in host memory to the device
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printArray(h_A , arr_size/(sizeof(float)));

        //Generating the output vector for the next iteration 
        free(h_sdata);
        sdata_size = num_blocks*(SIZE*SIZE)*sizeof(float);
        h_sdata = (float*)malloc(sdata_size);
        
        // Allocate the device output vector sdata
        err = cudaMalloc((void**)&d_sdata,sdata_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //Copying the new output vector to device
        printf("Coping output array from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_sdata, h_sdata,sdata_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array sdata from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }     

    }
    
    //float *result;
    /*{
    result = (float*)malloc(SIZE*SIZE*sizeof(float));
    size_t result_size = SIZE*SIZE*sizeof(float);
    printf("\n\nAnswer :- \n\n");
    //Copying back output array
    err = cudaMemcpy(result,d_sdata,result_size,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    }*/
    printf("Result :- \n");
    printArray(h_sdata , SIZE*SIZE);
    
}

void optimised4(const int warp_size,const int procs,const int max_threads_per_block){
    int n;
    int response;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t arr_size;

    printf("Enter no of Arrays : ");
    scanf("%d",&n);

    //Defining the matrices
    float* h_A;
    arr_size = (SIZE*SIZE)*n*sizeof(float);
    h_A = (float*)malloc(arr_size);
    fillArray(h_A,n);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**)&d_A,arr_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A  in host memory to the device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Defining the blocksize and the grid size
    int threads_per_block = warp_size*(max_threads_per_block/warp_size);
    int num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);

    num_blocks = ceil(num_blocks/2.0);
    
    //Defining h_sdata;
    float* h_sdata;
    size_t sdata_size = (SIZE*SIZE)*num_blocks*sizeof(float);
    h_sdata = (float*)malloc(sdata_size);

    // Allocate the device vector sdata 
    float *d_sdata = NULL;
    err = cudaMalloc((void**)&d_sdata,sdata_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printArray(h_A , arr_size/(sizeof(float)));
    
    //Main recursive kernel invocation
    while(1){

        //Kernel Invocation
        printf("Launched kernel with %d blocks and %d threads per block \n",num_blocks,threads_per_block);
        optimKernel4<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(d_A,d_sdata,n*(SIZE*SIZE));
        
        //Updation steps
        //.............

        //Copying back output array
        err = cudaMemcpy(h_sdata,d_sdata,sdata_size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy output array from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printArray(h_sdata,sdata_size/sizeof(float));

        //TERMINATION STATEMENT
        if(sdata_size == SIZE*SIZE*sizeof(float))
            break;  
        
        //Freeing unused memory
        free(h_A);
        err = cudaFree(d_A);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Freeing unused memory
        err = cudaFree(d_sdata);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //updation of n and num_blocks 
        n = num_blocks;
        num_blocks = ceil(n*(SIZE*SIZE)/(float)threads_per_block);
        
        //Generating input for the next iter.
        arr_size = sdata_size;
        h_A = h_sdata;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A,arr_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vector A  in host memory to the device
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A,arr_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printArray(h_A , arr_size/(sizeof(float)));

        //Generating the output vector for the next iteration 
        free(h_sdata);
        sdata_size = num_blocks*(SIZE*SIZE)*sizeof(float);
        h_sdata = (float*)malloc(sdata_size);
        
        // Allocate the device output vector sdata
        err = cudaMalloc((void**)&d_sdata,sdata_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector sdata (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //Copying the new output vector to device
        printf("Coping output array from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_sdata, h_sdata,sdata_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array sdata from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }     

    }
    
    //float *result;
    /*{
    result = (float*)malloc(SIZE*SIZE*sizeof(float));
    size_t result_size = SIZE*SIZE*sizeof(float);
    printf("\n\nAnswer :- \n\n");
    //Copying back output array
    err = cudaMemcpy(result,d_sdata,result_size,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    }*/
    printf("Result :- \n");
    printArray(h_sdata , SIZE*SIZE);
    
}


int main(){
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    double walltime;   
    int device_count;   
    cudaDeviceProp* prop;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nNumber of devices :- %d\n\n",device_count);
    printf("Device Properties :-\n");
    for(int  i = 0;i<device_count;i++){
        //printf("i : %d\n",i);
        err = cudaGetDeviceProperties(prop,i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get device properties (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printDevProp(prop);
    }
    int warp_size,procs,max_threads_per_block;
    warp_size = prop->warpSize;
    procs = prop->multiProcessorCount;
    max_threads_per_block = prop->maxThreadsPerBlock;

    //naiveImplementation(warp_size,procs,max_threads_per_block);
    //optimised1(warp_size,procs,max_threads_per_block);
    //optimised2(warp_size,procs,max_threads_per_block);
    optimised3(warp_size,procs,max_threads_per_block);
    //optimised4(warp_size,procs,max_threads_per_block);   ///DOES NOT WORK ATM


    return 0;
}

