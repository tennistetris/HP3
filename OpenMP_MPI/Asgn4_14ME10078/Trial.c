#include <stdio.h>
#include <stdlib.h>

int size = 32;

int main(int argc,char** argv){

    printf("Enter the size of the matrix : ");
    scanf("%d",&size);

    FILE* fp;
    fp=fopen("Input1.txt","w");
    //Printing the size of the matrix onto the file
    fprintf(fp,"%d %d\n",size,size);
    //Prinitng the data onto the file
    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            fprintf(fp,"%d ",i*size+j);
        }
        fprintf(fp,"\n");
    }
    printf("\nDone printing...\n");
    fclose(fp);

}