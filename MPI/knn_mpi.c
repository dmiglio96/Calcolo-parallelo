#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "check.h"
#include "input.h"
#include "knn_functions.h"
#include "utility.h"


int main (int argc, char* argv[]){

	int rank, size;                    				//rank identificativo del processo, size totale
    int root_rank = 0;                        //rank del processo root
    
    double time_init, final_time, elapsed_time;
    time_init = MPI_Wtime();

    bool checkresult = false;                  //effettuare controlli risulato con seriale
    bool saveData = true;                     //salvare risultati


    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	
      const char * trainFile = argv[1];
	const char * testFile = argv[2];
	if((argc -1) != 2){
		printf("Errore non sono stati specificati correttamente i file del dataset!\n");
		exit(EXIT_FAILURE);
	}
    
    //check consistenza valore K
	if (K > N){
		printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
		exit(EXIT_FAILURE);
	}

	if (K % 2 == 0){
		printf("Inserire un numero di vicini dispari!\n");
		exit(EXIT_FAILURE);
	}

	//numero di elementi del testing che ogni processo deve gestire per il calcolo LBP
    //serve per scatterv
    int countsRow[size]; 

    //indirizzi di inizio della porzione che ogni processo deve gestire 
    int displsRow[size]; 

    //numero di elementi che ogni processo deve gestire
    int countsClasses[size]; 

    //indirizzi di inizio della porzione che ogni processo deve gestire 
    int displsClasses[size];

    //funzione che permette di suddividere i record di testing da gestire
    suddivisioneTestingData(countsRow, displsRow, size);
    suddivisioneClasses(countsClasses, displsClasses, size);

	if(rank ==0){

		float * trainingData = (float *) malloc(N* M * sizeof(float));
		float * testingData = (float *) malloc(P* M * sizeof(float));

		uint8_t * classesTraining = (uint8_t*) malloc(N *sizeof(uint8_t));
		uint8_t * classesTesting = (uint8_t*)  malloc(P *sizeof(uint8_t));

		if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL){
			printf("Not enough memory!\n");
			exit(EXIT_FAILURE);
		}

		//read data file
		read_file(trainFile, N, M, trainingData, classesTraining);
		read_file(testFile, P, M, testingData, classesTesting);

	printf("nome file %s \n", trainFile);
	printf("nome file test %s \n", testFile);

		double pre_time = MPI_Wtime();
	    double tempo_passato = pre_time - time_init;
	    printf("Inizializzazione e lettura dati effetuata in %f  \n", tempo_passato);
		
		/*
		for(int i=0; i< size; i++){
			printf("displ %d count %d \n", displsRow[i], countsRow[i]);
			//printf("displ label %d count %d\n", displsClasses[i], countsClasses[i]);
		}

		for(int i=0; i< size; i++){
			//printf("displ %d count %d \n", displsRow[i], countsRow[i]);
			printf("displ label %d count %d\n", displsClasses[i], countsClasses[i]);
		}
		*/
	    
	    //invio dati di training
	    MPI_Bcast(trainingData, N*M, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
	    MPI_Bcast(classesTraining, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

	    //dati locali che il processo root deve gestire
		float * localTestingData = (float *) malloc(countsRow[rank] * sizeof(float));
		uint8_t * localClassesTesting = (uint8_t*)  malloc(countsRow[rank]/M *sizeof(uint8_t));	

		if(localTestingData == NULL || localClassesTesting == NULL){
			printf("Not enough memory!\n");
			exit(EXIT_FAILURE);
		}
	    //broadcast
		//Invio ad ogni processo la porzione di immagine a colori
   		MPI_Scatterv(testingData, countsRow, displsRow, MPI_FLOAT, localTestingData, countsRow[root_rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
   		

   		MPI_Scatterv(classesTesting, countsClasses, displsClasses, MPI_UINT8_T, localClassesTesting, countsClasses[root_rank], MPI_INT, root_rank, MPI_COMM_WORLD);
   		
		//printData(trainingData, classesTraining, N);
		//printData(testingData, classesTesting, P);

		//risultato delle classificazioni
   		int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);
   		if(confusionMatrix == NULL){
			printf("Not enough memory!\n");
			exit(EXIT_FAILURE);
		}

		//calcolo della matrice di confusione per la porzione che il processo deve
		int localError = 0;
		localError = calculatelocalKNN(trainingData, localTestingData, classesTraining, localClassesTesting, countsRow[rank]/M, confusionMatrix);
			
	    
	
		//printConfusionMatrix(confusionMatrix);
		int* final_CM = (int*) malloc(sizeof(int)* LABELS * LABELS);
		MPI_Reduce(confusionMatrix, final_CM, LABELS*LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

		//recupero errori locali ogni processo
		int errors = 0;
		MPI_Reduce(&localError, &errors, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

		//MPI_Barrier(MPI_COMM_WORLD);
		//printf("Matrice finale\n");
		//printConfusionMatrix(final_CM);

		//printf("Errori totali: %d\n", errors);
		//printf("Record corretti: %d accuratezza (%.2f%%)\n", P - errors, 100 - ((float) errors / P) * 100);
		
		
		if(checkresult == true){
			checkResultKNN(trainingData, testingData, classesTraining, classesTesting, final_CM);
		}
		
		
		//dealloco memoria 
		free(trainingData); trainingData = NULL;
	    free(testingData); testingData = NULL;
	    free(classesTraining); classesTraining = NULL;
		free(classesTesting); classesTesting = NULL;
		free(localTestingData); testingData = NULL;
		free(localClassesTesting); classesTesting = NULL;
		free(confusionMatrix); confusionMatrix = NULL;
		free(final_CM); final_CM = NULL;
	

		final_time = MPI_Wtime();
	    elapsed_time = final_time - time_init;
	    printf("Tempo totale esecuzione %f  \n", elapsed_time);
	     
	     
	     //save on file 
	    if (saveData == true)
	    	saveResultsOnFile(elapsed_time, size);
	}
		
	//gestione processo non root
	else
	 	{	
	 		float * trainingData = (float *) malloc(N* M * sizeof(float));
	 		uint8_t * classesTraining = (uint8_t*) malloc(N *sizeof(uint8_t));
			
			float * localTestingData = (float *) malloc(countsRow[rank] * sizeof(float));
			uint8_t * localClassesTesting = (uint8_t*)  malloc(countsClasses[rank] *sizeof(uint8_t));

			if(trainingData == NULL || localTestingData == NULL || localClassesTesting == NULL || classesTraining == NULL){
				printf("Not enough memory!\n");
				exit(EXIT_FAILURE);
			}	
	 		
	 		//Ricevo dati di training
	 		MPI_Bcast(trainingData, N*M, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
	    	MPI_Bcast(classesTraining, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

	    	//ricezione porzione di dati di testing da trattare
	    	MPI_Scatterv(NULL, NULL, NULL, NULL, localTestingData, countsRow[rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
   		
			MPI_Scatterv(NULL, NULL, NULL, NULL, localClassesTesting, countsClasses[rank], MPI_UINT8_T, root_rank, MPI_COMM_WORLD);
   			

			int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);
	   		if(confusionMatrix == NULL){
				printf("Not enough memory!\n");
				exit(EXIT_FAILURE);
			}

			int localError = 0;
			localError = calculatelocalKNN(trainingData, localTestingData, classesTraining, localClassesTesting, countsRow[rank]/M, confusionMatrix);
			//printf("local error no root %d \n", localError);

			//printConfusionMatrix(confusionMatrix);
			
			
   			MPI_Reduce(confusionMatrix, NULL, LABELS*LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

   			MPI_Reduce(&localError, NULL, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

   			//MPI_Barrier(MPI_COMM_WORLD);
   			//dealloco memoria
	 		free(trainingData); trainingData = NULL;
	 		free(classesTraining); classesTraining = NULL;
		    free(localTestingData); localTestingData = NULL;
			free(localClassesTesting); localClassesTesting = NULL;
			free(confusionMatrix); confusionMatrix = NULL;
	 	}

	//tutti i processi devono chiudere correttamente MPI
 	MPI_Finalize();
	return 0;

}
