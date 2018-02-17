#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define HI(num) (((num) & 0x0000FF00) << 8) 
#define LO(num) ((num) & 0x000000FF) 

typedef struct _PGMData {
    int row;
    int col;
    int max_gray;
    int *image;
}PGMData;

int *allocate_dynamic_matrix(int row, int col)
{
    int *ret_val;
    int i;
 
    ret_val = (int *)malloc(row*col*sizeof(int));
    if (ret_val == NULL) {
    
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    return ret_val;
}
 
void deallocate_dynamic_matrix(int *matrix)
{
    free(matrix);
}

/*for reading:*/
void readPGM(const char *file_name, PGMData *data)
{
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;
    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL) {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
    fscanf(pgmFile, "%d", &data->col);
    fscanf(pgmFile, "%d", &(data->row));
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);
 
    data->image = allocate_dynamic_matrix((data->row), data->col);
    if (data->max_gray > 255) {
        for (i = 0; i < (data->row); ++i) {
            for (j = 0; j < data->col; ++j) {
                hi = fgetc(pgmFile);
                lo = fgetc(pgmFile);
                data->image[i*(data->row) + j] = (hi << 8) + lo;
            }
        }
    }
    else {
        for (i = 0; i < (data->row); ++i) {
            for (j = 0; j < data->col; ++j) {
                lo = fgetc(pgmFile);
                data->image[i*(data->row)+j] = lo;
            }
        }
    }
 
    fclose(pgmFile);
 
}

/*and for writing*/
 
void writePGM(const char *filename, const PGMData *data)
{
    FILE *pgmFile;
    int i, j;
    int hi, lo;
 
    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }
 
    fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, (data->row));
    fprintf(pgmFile, "%d ", data->max_gray);
 
    if (data->max_gray > 255) {
        for (i = 0; i < (data->row); ++i) {
            for (j = 0; j < data->col; ++j) {
                hi = HI(data->image[i*(data->row) + j]);
                lo = LO(data->image[i*(data->row) + j]);
                fputc(hi, pgmFile);
                fputc(lo, pgmFile);
            }
 
        }
    }
    else {
        for (i = 0; i < (data->row); ++i) {
            for (j = 0; j < data->col; ++j) {
                lo = LO(data->image[i*(data->row) + j]);
                fputc(lo, pgmFile);
            }
        }
    }
 
    fclose(pgmFile);
    free(data->image);
}

void checkHistrogram(int hist[],int hist_correct[]){
    int counter = 0;
    
    for (int i = 0;i<255;i++){
        if(fabs(hist_correct[i] - hist[i])>1e-5){
            printf("Error at position [%d]. \nActual Value : %d\nYour Value : %d\n",i,hist_correct[i],hist[i]);
            break;
            counter = 1;
        }
    }

    if(counter == 0){
        printf("\nSuccess....Both the implementaions give the same result\n");
    }

}

void histCorrect(int *image,int * hist_correct,const int size){
    double wtime = omp_get_wtime();
    for(int i = 0;i<size;i++)
        for(int j = 0;j<size;j++)
            hist_correct[image[i*size + j]] +=1;
    wtime = omp_get_wtime() - wtime;
    printf("\nNon Parallel Implementation of Part 1: %lf s.\n",wtime);
}
    

void naiveImplementationPart2(int *image,const int size){
    
    int i,j,sum=0;
    float mean;
    unsigned long int start_x,start_y;
    int segment_size = 10;
    int segment_area = segment_size*segment_size;
    int max_strides = size/segment_size;

    double wtime = omp_get_wtime(); 
    for(i = 0;i<max_strides;i++){
        for(j = 0;j<max_strides;j++){
    

            start_x = i*segment_size;
            start_y = j*segment_size;

            for(int k = 0;k<segment_size;k++){
                for(int l = 0;l<segment_size;l++){
                    //printf("%d ",image[(start_x+k)*size + (start_y+l)]);
                    sum+=image[(start_x+k)*size + (start_y+l)];
                }
            }

            mean = (float)sum/segment_area;

            if(mean<256){
                for(int k = 0;k<segment_size;k++){
                    for(int l=0;l<segment_size;l++){
                        image[(start_x+k)*size+(start_y+l)] = 0;
                    }
                }
            }
        }
    }
    wtime = omp_get_wtime()-wtime;
    printf("\nNaive implementation of Part 2 took %lf s.\n",wtime);

}

void checkImage(int * image_copy,int * image,const int size){
    
    unsigned long int index;
    int counter = 0;

    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            index = ((long)i)*size + j;
            if((image_copy[index] - image[index])>1e-5){
                printf("Error at index %ld .\nActual Value should be %d \nYour implementation gives %d",index,image_copy[index],image[index]);
                counter = 1;
                break;
            }
        }
        if(counter)break;
    }
    
    if(counter == 0){
        printf("\nSuccess....Both the implementaions give the same result\n");
    }

}

int main(){
    
    srand(100);
    const int size = 2000; 
    //int *image = (int*)(malloc(size*size*sizeof(int)));
    int *image; //comment for random initialization
    //int *image_copy = (int*)(malloc(size*size*sizeof(int)));//required for checking
    int* image_copy;
    double wtime;
	int thread_num = omp_get_max_threads();
	omp_set_num_threads(thread_num);
    unsigned long int index;
    
    //Comment the following statements for random initialization
    //..........................................................
    PGMData data_read,temp;
    readPGM("Julia_IIM_6_circle.pgm",&data_read);
    image = data_read.image;
    readPGM("Julia_IIM_6_circle.pgm",&temp);
    image_copy = temp.image;
    //..........................................................

    //Initializing the matrix
    for(int i=0;i<size;i++){
        for(int j = 0;j<size;j++){
            index = (long)i*size+j;
            //image[index] = rand()%256;   //Uncomment to return to random initialization
            //image_copy[index] = image[index];
        }
    }

    int hist[256],hist_correct[256],hist_copy[256];
    
    for(int i = 0;i<256;i++){
        hist[i] = 0;
        hist_correct[i] = 0;
        hist_copy[i] = 0;
    }

    //PART 1

    histCorrect(image,hist_correct,size);//this value is used for checking the parallel implementation
    
    wtime = omp_get_wtime();
    int chunk = 2*ceil(size/thread_num);
    #pragma omp parallel firstprivate(hist_copy,index)
    {
        #pragma omp for collapse(2)
        for(int i = 0;i<size;i++){
            for(int j = 0;j<size;j++){
                index = ((long)i)*size + j;
                hist_copy[image[index]]=hist_copy[image[index]] + 1;
            }
        }

        #pragma omp critical
        for(int i = 0;i<256;i++){
            hist [i] +=hist_copy[i];
        }

    }
    wtime = omp_get_wtime() - wtime;
    printf("\nParallel Implementation of Part 1: %lf s.\n",wtime);

    checkHistrogram(hist,hist_correct);

    int i,j,sum=0;
    float mean;
    int start_x,start_y;
    const int segment_size = 10;
    int segment_area = segment_size*segment_size;
    int max_strides = size/segment_size;
    //PART 2
    wtime = omp_get_wtime();

    #pragma omp for collapse(2)     
    for(i = 0;i<max_strides;i++){
        for(j = 0;j<max_strides;j++){
            for(int k = 0;k<segment_size;k++){
                for(int l = 0;l<segment_size;l++){
                    sum+=image[((long)i*segment_size+k)*size + (j*segment_size+l)];
    
                }
            }
            
            mean = ((float)sum)/segment_area;

            if(mean<256){
                for(int k = 0;k<segment_size;k++){
                    for(int l=0;l<segment_size;l++){
                        image[((long)i*segment_size+k)*size+((long)j*segment_size+l)] = 0;
                    }
                }
            }
        }
    }
    wtime = omp_get_wtime()-wtime;
    printf("\nParallel implementation of Part 2 took %lf s.\n",wtime);
    
    //Still a tad slow,try doing better

    /*
    //A slower parallel implementation
    int sum_segment[max_strides][max_strides],sum_copy[max_strides][max_strides];
    for(int i = 0 ;i<max_strides;i++){
        for(int j = 0;j<max_strides;j++){
            sum_segment[i][j] = 0;
            sum_copy[i][j] = 0;
        }
    }
    int k,l;
    wtime = omp_get_wtime();
    #pragma omp parallel firstprivate(k,l,sum_copy)
    {
        
        #pragma omp for collapse(2) 
        for(int i = 0;i<size;i++){
            for(int j = 0;j<size;j++){
                k = i%segment_size;
                l = j%segment_size;
                sum_copy[k][l] +=image[i*size + j];
            }
        }

        #pragma omp for collapse(2)
        for(k=0;k<max_strides;k++){
            for(l=0;l<max_strides;l++){
                sum_segment[k][l] +=sum_copy[k][l];
            }
        }
       
        //The following part is wrong
        #pragma omp for collapse(2)
        for(int i = 0;i<max_strides;i++){
            for(int j = 0;j<max_strides;j++){
                if(sum_segment[i][j]<20*segment_area){
                    for(k = 0;k<segment_size;k++){
                        for(l = 0;l<segment_size;l++){
                            image[(i*segment_size+k)*size +(j*segment_size)+l] = 0;
                        }
                    }

                }
            }
        }
    }
    wtime = omp_get_wtime()-wtime;
    printf("\nParallel implementation of Part 2 took %lf s.\n",wtime);
    */

    naiveImplementationPart2(image_copy,size);
    checkImage(image_copy,image,size);
    writePGM("modified_file.pgm",&data_read);
    
    return 0;
}