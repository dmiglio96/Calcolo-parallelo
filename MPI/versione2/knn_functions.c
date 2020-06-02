#include <stdint.h> 
#include "knn_functions.h"
#include "input.h"
#include "utility.h"
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

void sort(float * distance, uint8_t * index, int numeroElementi){
    //index[0] = 0;
    for(int i =1; i <numeroElementi; ++i){
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


void sortLine(float * distance, float * distance2, uint8_t* label1, uint8_t* label2){
	/*
	for(int i=0; i<K; i++)
		printf("distanza %f label %d ", distance[i],label1[i]);
	printf("\n");

	for(int i=0; i<K; i++)
		printf("distanza %f label %d ", distance2[i],label2[i]);
	printf("\n");
	*/

	float unionDistance[K*2];
	uint8_t unionLabel[K*2];
	for(int i=0; i < K; i++){
		unionDistance[i] = distance[i];
		unionDistance[i+K] = distance2[i];
		unionLabel[i]= label1[i];
		unionLabel[i+K] = label2[i];
	}

	/*printf("UNITI\n");
	for(int i=0; i<K*2; i++)
		printf("distanza %f label %d ", unionDistance[i],  unionLabel[i]);
	printf("\n");
	*/

	sort(unionDistance, unionLabel, K*2);
	for(int i=0; i< K; i++)
	{
		distance[i] = unionDistance[i];
		label1[i] = unionLabel[i];
	}
	/*
	printf("Dopo sort\n");
	for(int i=0; i<K; i++)
		printf("distanza %f label %d ", distance[i], label1[i]);
	printf("\n\n");
	*/
}

void suddivisioneTrainingData(int* countsRow, int* displsRow, int size){
    int remRow = N % size;                  //resto della divisione intera
    int sumRow = 0;                         //righe già assegnate

    for (int i = 0; i < size; i++) {            //ciclo su ogni riga
        countsRow[i] = (N/size) * M;            //numero intero di righe ad ognuno
        if (remRow > 0) {                       //se qualche processo deve gestire più righe
            countsRow[i]= countsRow[i] + M;      //vengono assegnate in ordine in modo che siano contigue
            remRow--;
        }

        displsRow[i] = sumRow;

        sumRow += countsRow[i];
    }

}

void suddivisioneClasses(int* countsRow, int* displsRow, int size){
    int remRow = N % size;                  //resto della divisione intera
    int sumRow = 0;                         //righe già assegnate

    for (int i = 0; i < size; i++) {            //ciclo su ogni riga
        countsRow[i] = (N/size);            //numero intero di righe ad ognuno
        if (remRow > 0) {                       //se qualche processo deve gestire più righe
            countsRow[i]= countsRow[i] +1;      //vengono assegnate in ordine in modo che siano contigue
            remRow--;
        }

        displsRow[i] = sumRow;

        sumRow += countsRow[i];
    }

}

void calculatelocalKNN(float* testingData, float* localTrainingData, uint8_t* localClassesTraining , int NrecordTrain, uint8_t *local_k_label, float* local_k_distance){

	//distanza per ogni record del test con i record di train da computare
    float* k_distances = (float*) malloc(sizeof(float)* P* NrecordTrain); 
        
    //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
    uint8_t * k_labels = (uint8_t*) malloc(sizeof(uint8_t)* P *NrecordTrain);

        

    //check memory
    if (k_distances == NULL || k_labels == NULL){
        printf("Not enough memory!\n");
        exit(EXIT_FAILURE);
    }
        
        
    //per ogni sample del testing
    for(int i=0; i< P; i++){
    	//printf("Indice i %d  numero record train %d \n", i, NrecordTrain);
        //calcolo distanze con tutti i punti del training da trattare
        for(int j=0; j<NrecordTrain; j++){
        	//printf("j %d record train, scrivo in %d riga train inizia a %d mentre test a %d\n ", j, i * NrecordTrain+ j, j*M,i*M);
            float dist = computeDist(&localTrainingData[j*M], &testingData[i*M]);
            //printf("Distanza calcolata %f e scritta in %d\n", dist, i * NrecordTrain+ j );
            k_distances[i * NrecordTrain+ j] = dist;
            k_labels[i * NrecordTrain +j] = localClassesTraining[j];
        }


		//for(int k=0; k<K; k++)
		//	printf("Indirizzo %d  %f label %d ",i*NrecordTrain +k, k_distances[i*NrecordTrain +k], k_labels[i*NrecordTrain +k]);
		//printf("\n\n");
		
           
        //ordino i dati
        sort(&k_distances[i*NrecordTrain], &k_labels[i*NrecordTrain], NrecordTrain);

        /*printf("Dopo sort\n");
        for(int k=0; k<K; k++)
			printf("Indirizzo %d  %f label %d ", i*NrecordTrain +k, k_distances[i*NrecordTrain +k], k_labels[i*NrecordTrain +k]);
		printf("\n\n");
		*/

        //recupero i primi K vicini locali
        for(int k=0; k <K; k++){
        	local_k_distance[i*K +k] = k_distances[i*NrecordTrain +k];
        	local_k_label[i*K +k] = k_labels[i*NrecordTrain +k];

        }

        /*for(int k=0; k<K; k++)
			printf("Dopo %f label %d ",local_k_distance[i*K +k], local_k_label[i*K +k]);
        printf("\n\n");
        */
	}
	free(k_distances); k_distances = NULL; 
    free(k_labels); k_labels = NULL;
    //printf("finito qua\n");
}