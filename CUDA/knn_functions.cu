#include "input.h"
#include "knn_functions.h" 
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


//funzione kernel in cui ogni thread computa la distanza tra il proprio sample di test e tutti quelli del train
__global__ void computeDist_kernel(const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ dev_distances){//, int* dev_labels){
	//indice inizio riga
	
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
  	int idy = threadIdx.y+blockDim.y*blockIdx.y;
	
	//printf("cx cy %d %d\n", cx, cy);
	//check extra thread
	if(idx < N && idy <P){
		//printf("cx cy %d %d\n", cx, cy);
		//__shared__ float train[M];
		//trai
		//__shared__ float test[M];
		//__syncthreads();
		float sum = 0.f;
	    #pragma unroll
	    for (int d=0; d<M; ++d) {
	    	//__ldg(d_a + i)
	    	float x = dev_train[idx*M +d];  
	    	float y = dev_test[idy* M +d];
	    	//loat x =__ldg(dev_train + idx*M + d);
	    	//float y =__ldg(dev_test + idy*M + d);
	        float diff = x - y;
	        sum += diff * diff;
	    }
    //return 
		dev_distances[(idy *N) + idx] = sqrtf(sum);//distanceFunction(&dev_train[cy*M], &dev_test[cx*M]);
		//printf("%.2f \n", dev_distances[cx *N + cy]);
		//dev_labels[cx* N + cy] = cy;
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



  



