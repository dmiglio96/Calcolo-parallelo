#include "input.h" 
#include "knn_functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

__global__ void singleDistance(const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ distance){
	int id = threadIdx.x; 
	float local_distance = dev_train[id] - dev_test[id];
	local_distance = local_distance *local_distance;
	atomicAdd(distance, local_distance);
	//distance = distance +local_distance
}

//funzione kernel in cui ogni thread computa la distanza tra il proprio sample di test e tutti quelli del train
__global__ void computeDist_kernel(const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ dev_distances){//, int* dev_labels){
	//indice inizio riga
	
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
  	int idy = threadIdx.y+blockDim.y*blockIdx.y;
	
	//printf("cx cy %d %d\n", cx, cy);
	//check extra thread
	if(idx < N && idy <P){
	    singleDistance<<<1, M>>>(&dev_train[idx*M], &dev_test[idy* M], &dev_distances[(idy *N) + idx]);
		cudaDeviceSynchronize();
		dev_distances[(idy *N) + idx] = sqrtf(dev_distances[(idy *N) + idx]);
	}
}


__global__ void sort_kernel(float* __restrict__ dev_distances, int* __restrict__ dev_labels){
	
	//indice inizio riga
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	//printf(" %d ", index);
	//check extra thread
	if(index < P){
		dev_labels[index*K] = 0;
		#pragma unroll
		for(int i=1; i< N; i++){
			float distanzaCorrente = dev_distances[index*N+i];
        	int indiceCorrente = i;
        	//dev_labels[index*K+i] = i;
        	//printf("distanza corrente %f confronto con %f\n", distanzaCorrente, dev_distances[index*N+ K-1]);
			if( i >= K && distanzaCorrente >= dev_distances[index*N+ K-1]){
            	continue;
        	}
			
			int j = i;
        	if (j > K-1)
            	j = K-1;
        
        	while(j > 0 && dev_distances[index*N+ j-1] > distanzaCorrente){
            	dev_distances[index*N +j] = dev_distances[index*N+j-1];
            	dev_labels[index*K+j] = dev_labels[index*K+j-1];
            	--j;
        	}

        	dev_distances[index*N+j] = distanzaCorrente;
        	dev_labels[index*K+j] = indiceCorrente;	
		}
	}
}



  



