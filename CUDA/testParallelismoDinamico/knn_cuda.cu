#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "input.h"
#include "knn_functions.h"
#include "check.h"
#include "cudaError.h"
#include "utility.h"


int main(int argc, char* argv[]) {
	//salvare o meno il risultato su file
    bool saveData = true;
	bool checkresult = false;
	const char * trainFile = argv[1];
	const char * testFile = argv[2];
	
	//device
	int deviceIndex = 0;

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


	//numero di schede presenti
	int count;
	HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    
    //check esistenza scheda disponbile
    if(deviceIndex < count)
    {
        HANDLE_ERROR(cudaSetDevice(deviceIndex));
    }
    else
    {
        printf("Device non disponbile!\n");
        exit(EXIT_FAILURE);        
    }

    // proprietà della scheda video
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceIndex));

    //printf("M : %d Max threads per block: %d\n",M, prop.maxThreadsPerBlock );
	//printf("Max thread dimensions: (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
	//printf("Max grid dimensions: (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
	//printf("\n" );
    /*
    int maxthread;
    cudaDeviceGetAttribute(&maxthread, maxThreadsPerBlock);
    //Check sforamento numero di thread per blocco 
    if (BLOCK_SIZE * BLOCK_SIZE > maxthread){
    	printf("Errore, superato massimo numero di thread per blocco!\n");
    	exit(EXIT_FAILURE);
    }
    */


	// misurare il tempo di esecuzione
	cudaEvent_t start, stop, stopRead, stopSendData, primoStep, secondoStep;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventCreate( &stopRead ) );
	HANDLE_ERROR( cudaEventCreate( &stopSendData ) );
	HANDLE_ERROR( cudaEventCreate( &primoStep ) );
	HANDLE_ERROR( cudaEventCreate( &secondoStep ) );
	
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	float * trainingData= (float *) malloc(N* M * sizeof(float));
	float * testingData= (float *) malloc(P* M * sizeof(float));
	//HANDLE_ERROR( cudaHostAlloc( (void**)&trainingData, N*M * sizeof( *trainingData ), cudaHostAllocDefault ) );
	//HANDLE_ERROR( cudaHostAlloc( (void**)&testingData, P*M * sizeof( *testingData ), cudaHostAllocDefault ) );

	int * classesTraining = (int*) malloc(N *sizeof(int));
	int * classesTesting = (int*)  malloc(P *sizeof(int));

	float * dist = (float *) malloc(P* N * sizeof(float));
	//HANDLE_ERROR( cudaHostAlloc( (void**)&dist, P*M * sizeof( *dist ), cudaHostAllocDefault ) );
	
	
	if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}

	
	read_file(trainFile, N, M, trainingData, classesTraining);
	read_file(testFile, P, M, testingData, classesTesting);

	printf("nome file %s \n", trainFile);
	printf("nome file test %s \n", testFile);

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stopRead, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stopRead ) );
	
	float elapsedTimeRead;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, stopRead ) );
	printf( "Lettura dati eseguita in: %f \n", elapsedTimeRead/1000 );
  	
	// puntattori ai dati sul device
	float* dev_train; 
	
	float* dev_test;

	float* dev_dist;

	int* dev_label;
	
	
	// alloco memoria per il dataset sulla gpu in memoria globale
	HANDLE_ERROR( cudaMalloc( (void**)&dev_train, N * M * sizeof(float) ) );
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_test, P * M * sizeof(float) ) );

	//allocco matrice distanze e relative label
	HANDLE_ERROR( cudaMalloc( (void**)&dev_dist, P* N  * sizeof(float) ) );

	//HANDLE_ERROR( cudaMalloc( (void**)&dev_label, P * N * sizeof(int) ) );

	
	
	// copia elementi del dataset
	HANDLE_ERROR( cudaMemcpy( dev_train, trainingData, N * M * sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_test, testingData, P * M * sizeof(float), cudaMemcpyHostToDevice ) );	
	//HANDLE_ERROR( cudaMemcpy( dev_dist, dist, N * P * sizeof(float), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaEventRecord( stopSendData, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stopSendData ) );
	
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, stopSendData ) );
	printf( "Copia dati su GPU eseguita dopo : %f  secondi\n", elapsedTimeRead/1000 );
	
	//HANDLE_ERROR( cudaMemcpy( dev_label, label, N * P * sizeof(int), cudaMemcpyHostToDevice ) );

	// creo blocchi da BLOCK_SIZE * BLOCK_SIZE thread
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1); 

	//Numero di blocchi
	int dim_row = (P +1 % BLOCK_SIZE == 0) ? P / BLOCK_SIZE : P / BLOCK_SIZE + 1;
	int dim_col = (N + 1 % BLOCK_SIZE == 0) ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1;
	

	dim3 grid(dim_col, dim_row, 1); // a grid of CUDA thread blocks
	//printf("Numero di blocchi %d %d da %d \n", dim_row, dim_col, BLOCK_SIZE); 
	
	//cudaFuncSetCacheConfig(computeDist_kernel, cudaFuncCachePreferL1);
	// calcola distanza euclidea tra punti train e test
	computeDist_kernel<<<grid, block>>>(dev_train, dev_test, dev_dist);//, dev_label);

	int * label = (int*) malloc(P * K *sizeof(int));
	int* countsLabel = (int*) malloc(sizeof(int)* LABELS);
	int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);

	if(confusionMatrix ==NULL || countsLabel == NULL || label == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}

	// inizializza a zero la matrice di confusione
	initilizeArray(confusionMatrix, LABELS*LABELS, 0);
	
	// barriera per assicurarsi che tutte le distanze siano state calcolate
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaEventRecord(  primoStep, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(  primoStep ) );
	
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, primoStep ) );
	printf( "Distanze calcolate dopo : %f  secondi\n", elapsedTimeRead/1000 );
	

	//dealloco dataset su device non più utile
	
	HANDLE_ERROR( cudaFree(dev_train) );
    HANDLE_ERROR( cudaFree(dev_test) );
	
	//cudaDeviceSynchronize();	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_label, P * K * sizeof(int) ) );
	//HANDLE_ERROR( cudaMemcpy( dev_label, label, P*K * sizeof(int), cudaMemcpyHostToDevice ) );

	dim3 blockSort(BLOCK_SIZE, 1, 1);
	dim3 gridSort(dim_row, 1, 1);
	//printf("Numero di blocchi per il sort %d da %d \n", dim_row, BLOCK_SIZE); 
	sort_kernel<<<gridSort, blockSort>>>(dev_dist, dev_label);
	// barriera per assicurare che siano tutti ordinat

	cudaDeviceSynchronize();

	//recupero risultati dalla GPU
	//HANDLE_ERROR(cudaMemcpy(dist , dev_dist, P * N * sizeof(float), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR(cudaMemcpy(label , dev_label, P * K * sizeof(int), cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaEventRecord(  secondoStep, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(  secondoStep ) );
	
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, secondoStep ) );
	
	printf( "Ordinate e ricevute dopo : %f  secondi\n", elapsedTimeRead/1000 );
	
	/*printf("Dopoooooooo\n");
	for(int i=0; i < P; i++){
		for(int j=0; j < K; j++)
			printf(" %d ", label[i*K +j]);
		printf("\n\n");
	}
	*/
	
	
	
	// numero di errori di classificazione commessi dall'algoritmo KNN
	int error = 0;
	
	//il calcolo della matrice di confusione finale viene lasciato alla cpu
	for (int i=0; i<P; i++){
		initilizeArray(countsLabel, LABELS, 0);
		int bestLabel = 0;
		for(int j=0; j<K; j++){	
			int indice = label[i*K+j];
			int classe = classesTraining[indice]; 
			countsLabel[classe] = countsLabel[classe] + 1;
			if(countsLabel[classe] > countsLabel[bestLabel])
				bestLabel = classe;
			}

		int realLabel = classesTesting[i];
		if (realLabel != bestLabel){
			error = error + 1;
		}
			
		//update confusion matrix
		confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] +1;	
	}
	
	//stampa Confusion matrix
	//printConfusionMatrix(confusionMatrix);
	//printf("Errori totali: %d\n", error);
	//printf("Record corretti: %d accuratezza (%.2f%%); ", P - error, 100 - ((float) error / P) * 100);
	
	// controllo risultato con il seriale
	if(checkresult == true){
		checkResultKNN(trainingData, testingData, classesTraining, classesTesting, confusionMatrix);
	}	
	
	// dealloca memoria CPU

	//HANDLE_ERROR( cudaFreeHost( trainingData) );
	//HANDLE_ERROR( cudaFreeHost( testingData ) );
	//HANDLE_ERROR( cudaFreeHost( dist ) );
	
	free(trainingData); trainingData = NULL;
	free(testingData); testingData = NULL;
	free(dist); dist=NULL;
	
	free(classesTraining); classesTraining = NULL;
	free(classesTesting); classesTesting = NULL;
	
	free(confusionMatrix); confusionMatrix=NULL;
	
	free(label); label=NULL;
	free(countsLabel); countsLabel= NULL;
	
	//dealloco memoria GPU
    //HANDLE_ERROR( cudaFree(dev_train) );
    //HANDLE_ERROR( cudaFree(dev_test) );

    HANDLE_ERROR( cudaFree(dev_label ) );
    HANDLE_ERROR( cudaFree(dev_dist ) );
    	
	
	// conteggio tempo totale di esecuzione
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	
	printf( "Total time: %f \n", elapsedTime/1000 );
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	//HANDLE_ERROR( cudaEventDestroy( stopRead ) );

	//save on file
	if(saveData == true)
      saveResultsOnFile(elapsedTime/1000);

	return 0;
}
