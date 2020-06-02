#include "input.h" 
#include "knn_functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


//funzione kernel in cui ogni thread computa la distanza tra il proprio sample di test e tutti quelli del train
__global__ void computeDist_kernel(const float *  __restrict__ train, size_t pitchTrain, const float*  __restrict__ test, size_t pitchTest, float*  __restrict__ dist, size_t pitchDist)
{
   int tidx = blockIdx.x*blockDim.x + threadIdx.x;
   int tidy = blockIdx.y*blockDim.y + threadIdx.y;

   if ((tidx < N) && (tidy < P))
   {  
        float sum = 0.f;
        float *row_train = (float *)((char*)train + tidx * pitchTrain);
        float *row_test = (float *)((char*)test + tidy * pitchTest);
        #pragma unroll
        for (int d=0; d<M; ++d) {
          float x = row_train[d];
          float y = row_test[d];
          float diff = x - y;
          sum += diff * diff;
          //printf(" riga test %d riga train %d elemento %d Confronto train %.2f test %.2f \n", tidy, tidx, d, x, y);
        }
      //return 
      float *row_dist = (float *)((char*)dist + tidy * pitchDist);
      row_dist[tidx] = sqrt(sum);
      //dist[(idx *P) + idy] = sqrt(sum);




       /*
       if(tidx == 0 && tidy == 0){
          printf("\nTrain\n");
          for(int i =0; i<N; i++){
            float *row_a = (float *)((char*)train + i * pitchTrain);
            for(int j=0; j< M; j++)
                printf(" %.2f ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            printf("\n");
        }

        printf("\ntest\n");
        for(int i =0; i<P; i++){
            float *row_a = (float *)((char*)test + i * pitchTest);
            for(int j=0; j< M; j++)
                printf(" %.2f ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            printf("\n");
        }

        printf("\n Distanze\n");
        for(int i =0; i<P; i++){
            float *row_a = (float *)((char*)dist + i * pitchDist);
            for(int j=0; j< N; j++)
                printf(" %.2f ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            printf("\n");
        }


       }
       */

       
    }
}


__global__ void sort_kernel(float*  __restrict__ dev_distances, size_t pitchDist, int*  __restrict__ dev_labels, size_t pitchLabel){
	
	//indice inizio riga
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	//printf(" %d ", index);
	//check extra thread
	if(index < P){
		/*if (index == 0){
			printf("\n Distanze\n");
        	for(int i =0; i<P; i++){
            	float *row_a = (float *)((char*)dev_distances + i * pitchDist);
            	for(int j=0; j< N; j++)
                	printf(" %.2f ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            	printf("\n");
        	}
		}
		*/
		int *row_label = (int *)((char*)dev_labels + index * pitchLabel);
		row_label[0] = 0;
		#pragma unroll
		for(int i=1; i< N; i++){
			int *row_label_loop = (int *)((char*)dev_labels + index * pitchLabel);
			
			float *row_dist = (float *)((char*)dev_distances + index * pitchDist);
			
			float distanzaCorrente = row_dist[i];
        	int indiceCorrente = i;
        	//dev_labels[index*K+i] = i;
        	//printf("distanza corrente %f confronto con %f\n", distanzaCorrente, dev_distances[index*N+ K-1]);
			if( i >= K && distanzaCorrente >= row_dist[K-1]){
            	continue;
        	}
			
			int j = i;
        	if (j > K-1)
            	j = K-1;
        
        	while(j > 0 && row_dist[j-1] > distanzaCorrente){
            	row_dist[j] = row_dist[j-1];
            	row_label_loop[j] = row_label_loop[j-1];
            	--j;
        	}

        	row_dist[j] = distanzaCorrente;
        	row_label_loop[j] = indiceCorrente;	
		}

		/*if(index == 0){
			if (index == 0){
			printf("\n Distanze dopo\n");
        	for(int i =0; i<P; i++){
            	float *row_a = (float *)((char*)dev_distances + i * pitchDist);
            	for(int j=0; j< K; j++)
                	printf(" %.2f ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            	printf("\n");
        	}

        	printf("\n label \n");
        	for(int i =0; i<P; i++){
            	int *row_a = (int *)((char*)dev_labels + i * pitchLabel);
            	for(int j=0; j< K; j++)
                	printf(" %d ",row_a[j]);// = row_a[tidx] * tidx * tidy;
            	printf("\n");
        	}
		}

		}
		*/

	}
}



  



