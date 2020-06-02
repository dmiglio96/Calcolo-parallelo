#include <stdint.h> 
#include "check.h"
#include "input.h"
#include "math.h"

void sortSerial(float * distance, int * index){
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

void checkResultKNN(float* trainingData, float* testingData, uint8_t* classesTraining, uint8_t* classesTesting, int* confusionMatrix){
    float* k_distances = (float*) malloc(sizeof(float)* N); 
    
    //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
    int * k_labels = (int*) malloc(sizeof(int) * N);

    int* serialCM = (int*) malloc(sizeof(int)* LABELS * LABELS);
    
    //label di ogni sample per majority voting
    int* countsLabel = (int*) malloc(sizeof(int)* LABELS);

    //check memory
    if (serialCM == NULL || countsLabel == NULL || k_distances == NULL || k_labels == NULL){// || distancesNeighbors == NULL){
        printf("Not enough memory!\n");
        exit(EXIT_FAILURE);
    }
    
    //inizializza a zero la matrice di confusione
    initilizeArray(serialCM, LABELS*LABELS, 0);
    
    //numero di errori compessi dall'algoritmo KNN
    int errors = 0;
    
    //per ogni sample del testing
    for(int i=0; i< P; i++){
        //calcolo distanze con tutti i punti del training
        for(int j=0; j<N; j++){
            k_distances[j] = computeDist(&trainingData[j*M], &testingData[i*M]);
            k_labels[j] = j;
        }
        
        //ordino i dati
        sortSerial(k_distances, k_labels);
        
        
        //inizializza a zero il vettore
        initilizeArray(countsLabel, LABELS, 0);
        int bestLabel = 0;
        
        //per i primi k vicini
        for(int j=0; j<K; j++){ 
            //int indice = k_labels[j];
            int indice = k_labels[j];
            int label = classesTraining[indice];
            countsLabel[label] = countsLabel[label] + 1;
            if(countsLabel[label] > countsLabel[bestLabel])
                bestLabel = label;
        }


        int realLabel = classesTesting[i];
        //printf("label %d real label %d\n", bestLabel, realLabel);
        if (realLabel != bestLabel)
            errors = errors + 1;
        
        //update confusion matrix
        serialCM[realLabel * LABELS + bestLabel] = serialCM[realLabel * LABELS + bestLabel] +1;
        
    }
    
    bool error = false; 
    for(int i=0; i <LABELS; i++){
        for(int j=0; j< LABELS; j++){
            if(serialCM[i*LABELS + j] != confusionMatrix[i*LABELS +j]){
                printf("Errore indice [%d, %d] valore seriale %d valore parallelo %d\n", i, j, serialCM[i*LABELS + j], confusionMatrix[i*LABELS + j]);
                error = true;
            }
        }

    }
    if(error == false){
        printf("Operazione eseguita con successo\n");
    }

    //printConfusionMatrix(serialCM);

    free(countsLabel); countsLabel = NULL; 
    free(k_distances); k_distances = NULL; 
    free(k_labels); k_labels = NULL;
    free(serialCM); serialCM = NULL;
}