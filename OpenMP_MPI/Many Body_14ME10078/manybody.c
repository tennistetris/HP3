#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

const int NUM_SPHERES = 1000;
const int XDIM = 100;
const int YDIM = 200;
const int ZDIM = 400;
const int SEED = 42;
const int TIME = 3600;
const float DELTA_T =0.005;
const double G = 6.67408e-11;
const float MASS = 1;
const float DIAMETER = 1;
const double INFINITESMAL = 1e-4;

struct position{
    float x;
    float y;
    float z;
};

struct velocity{
    double x;
    double y;
    double z;
};

struct acceleration{
    double x;
    double y;
    double z;
};

struct particle{
    struct position pos;
    struct velocity vel;
    struct acceleration acc;    
};



double calculateForceNaive(struct particle* particles,int i,int j){
    
    struct position p1,p2;
    p1 = particles[i].pos;
    p2 = particles[j].pos;
    float mass1,mass2;
    mass1 = MASS;mass2 = MASS;
    double dist = sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2))+INFINITESMAL;
    //printf("DISTANCE[%d][%d] : %lf",i,j,dist); 
    double force = G*mass1*mass2/pow(dist,2);
    return force;
}

void naiveImplementation(struct particle* particles){
    //int steps =(double)TIME/DELTA_T;
    //printf("\nSTEPS  : %d\n",steps);
    FILE * fp;
    fp = fopen("log.txt","w");
    double wtime;
    unsigned int steps = 72000;
    for(int i = 0;i<steps;i++){
        //printf("\nITERATION NO : %lld\n",i);

        wtime = omp_get_wtime();

        //#pragma omp parallel
        {   
            #pragma omp for 
            for(int j = 0;j<NUM_SPHERES;j++){
                particles[j].vel.x += particles[j].acc.x*DELTA_T/(2*MASS);
                particles[j].vel.y += particles[j].acc.y*DELTA_T/(2*MASS);
                particles[j].vel.z += particles[j].acc.z*DELTA_T/(2*MASS);
            }
            
            #pragma omp barrier

            #pragma omp for
            for(int j = 0;j<NUM_SPHERES;j++){
                particles[j].pos.x += particles[j].vel.x*DELTA_T;
                particles[j].pos.y += particles[j].vel.y*DELTA_T;
                particles[j].pos.z += particles[j].vel.z*DELTA_T;       
            }
            
            #pragma omp for    
            for(int j = 0;j<NUM_SPHERES;j++){
                particles[j].acc.x = 0;
                particles[j].acc.y = 0;
                particles[j].acc.z = 0;
            }

            #pragma omp barrier
            
            double dist,force;
            struct position p1,p2,dir;
            #pragma omp for private(p1,p2,dir,dist,force)
            for(int j = 0;j<NUM_SPHERES;j++){
                for(int k=0;k<NUM_SPHERES;k++){
                    if(j!=k)
                        force = calculateForceNaive(particles,j,k);//Scalar quantity always positive
                    else 
                        force = 0;
                    //printf("Force[%d][%d]  : %.14lf \n",j,k,force);
                    //struct position p1,p2,dir;
                    p1 = particles[k].pos;
                    p2 = particles[j].pos;
                    dist = sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2))+INFINITESMAL; 
                    dir.x = (p1.x-p2.x)/dist;
                    dir.y = (p1.y-p2.y)/dist;
                    dir.z = (p1.z - p2.z)/dist;

                    #pragma omp critical
                    {
                        particles[k].acc.x -= (force/MASS)*dir.x;
                        particles[k].acc.y -= (force/MASS)*dir.y;
                        particles[k].acc.z -= (force/MASS)*dir.z;
                    }
                    particles[j].acc.x += (force/MASS)*dir.x;
                    particles[j].acc.y += (force/MASS)*dir.y;
                    particles[j].acc.z += (force/MASS)*dir.z;
                }
            }
            
            #pragma omp barrier

            #pragma omp for
            for(int j = 0;j<NUM_SPHERES;j++){
                particles[j].vel.x += particles[j].acc.x*DELTA_T/(2*MASS);
                particles[j].vel.y += particles[j].acc.y*DELTA_T/(2*MASS);
                particles[j].vel.z += particles[j].acc.z*DELTA_T/(2*MASS);
            }

            #pragma omp barrier
            
            //CODE FOR REFLECTION
            
            #pragma omp for
            for(int j = 0;j<NUM_SPHERES;j++){

                if(particles[j].pos.x<0)
                    particles[j].pos.x = - particles[j].pos.x;
                if(particles[j].pos.y<0)
                    particles[j].pos.y = - particles[j].pos.y;
                if(particles[j].pos.z<0)
                    particles[j].pos.z = - particles[j].pos.z;
                
                if(particles[j].pos.x>XDIM)
                    particles[j].pos.x -= XDIM;
                if(particles[j].pos.y>YDIM)
                    particles[j].pos.y -= YDIM;
                if(particles[j].pos.z>ZDIM)
                    particles[j].pos.z -= ZDIM;
                
            }
            
            #pragma omp barrier
            
        }

        //printf("\nITERATION NO : %d\n",i);
        if(!((int)i%100)){
            //printf("\nITERATION NO : %d\n",i);
            for(int j = 0 ;j<NUM_SPHERES;j++){
                printf("%lf ",particles[j].pos.x);
                printf("%lf ",particles[j].pos.y);
                printf("%lf ",particles[j].pos.z);
            }
            printf("\nTERMINATE\n");
        }
        wtime = omp_get_wtime()-wtime;
        fprintf(fp,"Time taken for iteration[%d] : %lf \n",i,wtime);
    } 
    fclose(fp);
}

int main(){
    struct particle *particles;
    srand(SEED);
    double wtime;
	int thread_num = omp_get_num_procs();
	omp_set_num_threads(thread_num);
    particles = (struct particle*)malloc(NUM_SPHERES*sizeof(struct particle));

    //printf("GRAVITATIONAL CONSTANT : %.14lf\n",G);
    //INITIALIZING
    for(int i = 0;i<NUM_SPHERES;i++){
        scanf("%f",&particles[i].pos.x);
        scanf("%f",&particles[i].pos.y);
        scanf("%f",&particles[i].pos.z);
        scanf("%lf",&particles[i].vel.x);
        scanf("%lf",&particles[i].vel.y);
        scanf("%lf",&particles[i].vel.z);
        scanf("%lf",&particles[i].acc.x);
        scanf("%lf",&particles[i].acc.y);
        scanf("%lf",&particles[i].acc.z);
    }

    naiveImplementation(particles);
    
    return 0;
}