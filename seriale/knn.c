#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "input.h"
#include "knn_functions.h"
#include "utility.h"


int main (int argc, char* argv[]){

	//star timer
  	clock_t start, time1, time2, time3; 
  	start = clock();
	//salvare o meno il risultato su file
    	bool saveData = true;
	const char * trainFile = argv[1];
	const char * testFile = argv[2];
	if((argc -1) != 2){
		printf("Errore non sono stati specificati correttamente i file del dataset!\n");
		exit(EXIT_FAILURE);
	}

	if (K > N){
		printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
		exit(EXIT_FAILURE);
	}

	if (K % 2 == 0){
		printf("Inserire un numero di vicini dispari!\n");
		exit(EXIT_FAILURE);
	}
	
	//printf("M: %d \n",M); 
	float * trainingData = (float *) malloc(N* M * sizeof(float));
	float * testingData = (float *) malloc(P* M * sizeof(float));

	int * classesTraining = (int*) malloc(N *sizeof(int));
	int * classesTesting = (int*)  malloc(P *sizeof(int));

	//distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini)
    float* k_distances = (float*) malloc(P* N *sizeof(float)); 
    
    //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
    int * k_labels = (int*) malloc(P * N * sizeof(int));

	if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL || k_distances == NULL || k_labels == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}
	

	read_file(trainFile, N, M, trainingData, classesTraining);
	read_file(testFile, P, M, testingData, classesTesting);
	printf("nome file %s \n", trainFile);
	printf("nome file test %s \n", testFile);

	time1 = clock();
    float t = (float)(time1 - start)/ (float) CLOCKS_PER_SEC;
    printf("Finish read time in: %f \n", t);
	
	//printData(trainingData, classesTraining, N);
	
	//printf("\n\n");
	//printData(testingData, classesTesting, P);
	

	int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);
	
	//label di ogni sample per majority voting
	int* countsLabel = (int*) malloc(sizeof(int)* LABELS);

	//check memory
	if (confusionMatrix == NULL || countsLabel == NULL ){// || distancesNeighbors == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}
	
	//inizializza a zero la matrice di confusione
	initilizeArray(confusionMatrix, LABELS*LABELS, 0);
	
	//numero di errori compessi dall'algoritmo KNN
	int error = 0;
	
	//per ogni sample del testing
	for(int i=0; i< P; i++){
		//calcolo distanze con tutti i punti del training
		for(int j=0; j<N; j++){
			k_distances[i*P+j] = computeDist(&trainingData[j*M], &testingData[i*M]);
			k_labels[i*P +j] = classesTraining[j];
			//printf("Classe train %d \n", k_labels[i*P +j]);
		}
		
		//ordino i dati
		//printf("label prima di sort %d ", k_labels[i*P]);
		sort(&k_distances[i*P], &k_labels[i*P]);
		//printf(" label dopo sort %d\n", k_labels[i*P]);
		
		//inizializza a zero il vettore
		initilizeArray(countsLabel, LABELS, 0);
		int bestLabel = 0;
		
		//per i primi k vicini
		for(int j=0; j<K; j++){	
			//int indice = k_labels[j];
			//int label = classesTraining[indice]; 
			int label = k_labels[i *P +j];
			//printf(" label %d ", label);
			countsLabel[label] = countsLabel[label] + 1;
			if(countsLabel[label] > countsLabel[bestLabel])
				bestLabel = label;
		}

		int realLabel = classesTesting[i];
		if (realLabel != bestLabel)
			error = error + 1;
		
		//update confusion matrix
		//printf("real label %d label %d indice %d\n", realLabel-1, bestLabel-1, (realLabel-1) * LABELS + (bestLabel-1));
		confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] +1;
		
	}
	

		
	//printConfusionMatrix(confusionMatrix);

	//printf("Errori totali: %d\n", error);
	//printf("Record corretti: %d accuratezza (%.2f%%); \n", P - error, 100 - ((float) error / P) * 100);

	
	free(trainingData); trainingData = NULL;
    free(testingData); testingData = NULL;
    
    free(confusionMatrix); confusionMatrix = NULL;
    free(countsLabel); countsLabel = NULL;
    free(k_distances); k_distances = NULL;
    free(k_labels); k_labels = NULL;

    free(classesTraining); classesTraining = NULL;
	free(classesTesting); classesTesting = NULL;

	
	//calcualte time
    float totaltime = (float)(clock() - start)/ (float) CLOCKS_PER_SEC;
    printf("total time: %f \n", totaltime);
     
     
     //save on file 
    if (saveData == true)
      saveResultsOnFile(totaltime);
	

	return 0;
}
