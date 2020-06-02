#ifdef _OPENMP
#include <omp.h> 
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "input.h"
#include "knn_functions.h"
#include "utility.h"
#include "check.h"


int main (int argc, char* argv[]){
	//salvare o meno il risultato su file
    bool saveData = true;

    bool checkresult = false;

    //star timer
    #ifdef _OPENMP 
      double start = omp_get_wtime();
      double time, time1;
    #else
      clock_t time, time1;
      time_t start = clock();
    #endif

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

	const char * trainFile = argv[1];
	const char * testFile = argv[2];

	float * trainingData = (float *) malloc(N* M * sizeof(float));
	float * testingData = (float *) malloc(P* M * sizeof(float));

	int * classesTraining = (int*) malloc(N *sizeof(int));
	int * classesTesting = (int*)  malloc(P *sizeof(int));

	if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}

	//reading data from file
	read_file(trainFile, N, M, trainingData, classesTraining);
	read_file(testFile, P, M, testingData, classesTesting);

	printf("nome file %s \n", trainFile);
	printf("nome file test %s \n", testFile);

	#ifdef _OPENMP
      time1 = omp_get_wtime();
      double t = time1 - start;

    #else
      time1 = clock(); 
      double t = (time1 - start)/ (double) CLOCKS_PER_SEC;
    #endif
    printf("Finish read in: %f \n", t);
	
	//printData(trainingData, classesTraining, N);
	//printData(testingData, classesTesting, P);
	

    //distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini)
    float* k_distances = (float*) malloc(sizeof(float)* P* N); 
    
    //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
    int * k_labels = (int*) malloc(sizeof(int) * P* N);

	


	//check memory
	if (k_distances == NULL || k_labels == NULL){// || distancesNeighbors == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}
	

	#ifdef _OPENMP
    	omp_set_num_threads(NT);
  	#endif 
	
	#pragma omp parallel default(none), shared(trainingData, testingData, classesTesting, classesTraining, k_distances,k_labels)
    {
		//per ogni sample del testing
		#pragma omp for collapse(2) schedule(guided, 1)// default(none), shared(trainingData, testingData,k_distances,k_labels)
		for(int i=0; i< P; i++){	
			//calcolo distanze con tutti i punti del training
			for(int j=0; j<N; j++){
				k_distances[i*N +j] = computeDist(&trainingData[j*M], &testingData[i*M]);
				k_labels[i*N + j] = j;
			}
		}
	
		//ordino ogni distanza
		#pragma omp for schedule(guided, 1)
		for(int i=0; i< P; i++){
			sort(&k_distances[i*N], &k_labels[i*N]);
		}
	
		//tutte distanze sono ordinate
		
	}
	
	
	//inizializza a zero la matrice di confusione
	int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);
	initilizeArray(confusionMatrix, LABELS*LABELS, 0);
	//numero di errori comessi dall'algoritmo KNN

	int* confusionMatrixPrivate = (int*) malloc(sizeof(int)* LABELS * LABELS * NT);
	
	

	

    //printf("thread numero %d\n", indexThread);

	int error = 0;

	#pragma omp parallel default(none) shared(classesTraining, classesTesting, k_labels, confusionMatrix, confusionMatrixPrivate, error)
	{	
		//numero di thread
		const int nthreads = omp_get_num_threads();
    	
    	//id del thread
    	const int indexThread = omp_get_thread_num();
		
		// un solo thread inizializza la matrice privata di appoggio
		#pragma omp single
		{	
			//printf("thread numero %d inizializza\n", indexThread);
			initilizeArray(confusionMatrixPrivate, LABELS*LABELS * NT, 0);
		}

		//printConfusionMatrix(&confusionMatrixPrivate[indexThread* LABELS*LABELS]);

		#pragma omp for schedule(guided, 1) reduction(+:error) 
		for (int i=0; i<P; i++){
			
			//label di ogni sample per majority voting
			int* countsLabel = (int*) malloc(sizeof(int)* LABELS);
			initilizeArray(countsLabel, LABELS, 0);

			int bestLabel = 0;
			for(int j=0; j<K; j++){	
				int indice = k_labels[i*N+j];
				
				int label = classesTraining[indice]; 
				//printf("indice %d label %d\n", indice, label);
				countsLabel[label] = countsLabel[label] + 1;
				if(countsLabel[label] > countsLabel[bestLabel])
					bestLabel = label;
			}

			int realLabel = classesTesting[i];
			if (realLabel != bestLabel){
				error += 1;
			}
			
			//update confusion matrix
			confusionMatrixPrivate[indexThread* LABELS*LABELS + (realLabel * LABELS + bestLabel)] = confusionMatrixPrivate[indexThread* LABELS*LABELS +(realLabel * LABELS + bestLabel)] +1;
			//printConfusionMatrix(&confusionMatrixPrivate[indexThread* LABELS*LABELS]);
			int indice = indexThread* LABELS*LABELS + (realLabel * LABELS + bestLabel);
			//printf("thread id %d scrive in %d\n", indexThread, indice);
    		free(countsLabel); countsLabel = NULL;
			
		}
		/*stampa di debug
		#pragma omp single
		{	
			printf("stampo tutto\n");
			for(int k=0;k<NT; k++){
				for(int i=0; i <LABELS; i++){
					for(int j=0; j < LABELS; j++)
						printf("%d ", confusionMatrixPrivate[k*LABELS*LABELS +(i* LABELS + j)]);
					printf("\n");
				}
			}
		}
		*/
		//calcolo matrice di confusione finale
		#pragma omp for
        for(int i=0; i<LABELS*LABELS; ++i) {
        	for(int thread=0; thread< nthreads; thread++){
            	confusionMatrix[i] = confusionMatrix[i] + confusionMatrixPrivate[LABELS*LABELS* thread + i];
        	}
        }

        
	}
		
					
	//printf("\n risultato \n");	
	//printConfusionMatrix(confusionMatrix);
	//printf("Errori totali: %d\n", error);
	//printf("Record corretti: %d accuratezza (%.2f%%); \n", P - error, 100 - ((float) error / P) * 100);

	if(checkresult == true){
		checkResultKNN(trainingData, testingData, classesTraining, classesTesting, confusionMatrix);
	}

	

	free(trainingData); trainingData = NULL;
    free(testingData); testingData = NULL;
    free(confusionMatrix); confusionMatrix = NULL;
    free(k_distances); k_distances = NULL;
    free(k_labels); k_labels = NULL;
    free(classesTraining); classesTraining = NULL;
	free(classesTesting); classesTesting = NULL;
	free(confusionMatrixPrivate); confusionMatrixPrivate = NULL;
	
	
	//calcualte time
    #ifdef _OPENMP
      time = omp_get_wtime() - start;
   	#else
      time = (clock - time)/ (double) CLOCKS_PER_SEC;
   	#endif
    printf("total time: %f \n\n", time);
     
     
     //save on file 
    if (saveData == true)
      saveResultsOnFile(time);
	


	return 0;
}
