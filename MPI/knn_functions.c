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

void suddivisioneTestingData(int* countsRow, int* displsRow, int size){
    int remRow = P % size;                  //resto della divisione intera
    int sumRow = 0;                         //righe già assegnate

    for (int i = 0; i < size; i++) {            //ciclo su ogni riga
        countsRow[i] = (P/size) * M;            //numero intero di righe ad ognuno
        if (remRow > 0) {                       //se qualche processo deve gestire più righe
            countsRow[i]= countsRow[i] + M;      //vengono assegnate in ordine in modo che siano contigue
            remRow--;
        }

        displsRow[i] = sumRow;

        sumRow += countsRow[i];
    }

}

void suddivisioneClasses(int* countsRow, int* displsRow, int size){
    int remRow = P % size;                  //resto della divisione intera
    int sumRow = 0;                         //righe già assegnate

    for (int i = 0; i < size; i++) {            //ciclo su ogni riga
        countsRow[i] = (P/size);            //numero intero di righe ad ognuno
        if (remRow > 0) {                       //se qualche processo deve gestire più righe
            countsRow[i]= countsRow[i] +1;      //vengono assegnate in ordine in modo che siano contigue
            remRow--;
        }

        displsRow[i] = sumRow;

        sumRow += countsRow[i];
    }

}

int calculatelocalKNN(float* trainingData, float* localTestingData, uint8_t* classesTraining, uint8_t* localClassesTesting, int NrecordTesting, int * localResultCM){       //distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini)
        float* k_distances = (float*) malloc(sizeof(float)* N); 
        
        //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
        int * k_labels = (int*) malloc(sizeof(int) * N);

        //label di ogni sample per majority voting
        int* countsLabel = (int*) malloc(sizeof(int)* LABELS);

        //check memory
        if (countsLabel == NULL || k_distances == NULL || k_labels == NULL){
            printf("Not enough memory!\n");
            exit(EXIT_FAILURE);
        }
        
        //inizializza a zero la matrice di confusione
        initilizeArray(localResultCM, LABELS*LABELS, 0);
        
        //numero di errori compessi dall'algoritmo KNN
        int error = 0;
        
        //per ogni sample del testing da trattare
        for(int i=0; i< NrecordTesting; i++){
            //calcolo distanze con tutti i punti del training
            for(int j=0; j<N; j++){
                k_distances[j] = computeDist(&trainingData[j*M], &localTestingData[i*M]);
                k_labels[j] = j;
            }
            
            //ordino i dati
            sort(k_distances, k_labels);
            
            
            //inizializza a zero il vettore
            initilizeArray(countsLabel, LABELS, 0);
            int bestLabel = 0;
            
            //per i primi k vicini
            for(int j=0; j<K; j++){ 
                int indice = k_labels[j];
                int label = classesTraining[indice]; 
                countsLabel[label] = countsLabel[label] + 1;
                if(countsLabel[label] > countsLabel[bestLabel])
                    bestLabel = label;
            }

            int realLabel = localClassesTesting[i];
            if (realLabel != bestLabel)
                error = error + 1;
            
            //update confusion matrix
            localResultCM[realLabel * LABELS + bestLabel] = localResultCM[realLabel * LABELS + bestLabel] +1;

        }

        free(countsLabel); countsLabel = NULL; 
        free(k_distances); k_distances = NULL; 
        free(k_labels); k_labels = NULL;
        return error;
}