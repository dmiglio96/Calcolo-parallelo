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
        int indiceCorrente = index[i];

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

/*
float calculateEuclideanDistance(float* testingData, float* trainingData, int size){

    float distanza = 0.0;
    float temp = 0.0;
    for(int i= 0; i< size; i++){
        temp = testingData[i] - trainingData[i];
        distanza = distanza + temp*temp;
    }

    return sqrt(distanza);
}

void updateNeightbors(struct distAndLabel * distancesNeighbors, float distanza, int classe){
    //printf("check\n");
    //printf("Distanza attuale %f testo con %f\n", distancesNeighbors[K-1].distance, distanza);
    if(distanza < distancesNeighbors[K-1].distance){
        //printf("aggiornamento indice %d \n", K-1);
        //aggiornamento
        distancesNeighbors[K-1].distance = distanza;
        distancesNeighbors[K-1].label = classe;

        //sort K best vicini fino a questo momento
        qsort(distancesNeighbors, K, sizeof(struct distAndLabel), compare);
        //printf("Dopo ordinamento:\n");
        //for(int i =0; i <K; i++)
        //  printf(" %f ", distancesNeighbors[i].distance);
        //printf("\n");
    }
}
*/
