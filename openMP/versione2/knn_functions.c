#include "knn_functions.h"
#include "input.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



float computeDist(float* train, float* test){
    float sum = 0.f;
    for (int d=0; d<M; ++d) {
        const float diff = train[d] - test[d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

void sort(float * distance, int * index){
    index[0] = 0;
    for(int i =1; i <N; ++i){
        float distanzaCorrente = distance[i];
        int indiceCorrente = i;

        if( i >= K && distanzaCorrente >= distance[K-1]){
            continue;
        }

        int j = i;
        if (j > K-1)
            j = K-1;
        
        while(j > 0 && distance[j-1] > distanzaCorrente){
            distance[j] = distance[j-1];
            index[j] = index[j-1];
            --j;
        }

        distance[j] = distanzaCorrente;
        index[j] = indiceCorrente;
    }
}