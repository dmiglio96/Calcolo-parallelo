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

    //funzione che permette di suddividere i record di train da gestire
    suddivisioneTrainingData(countsRow, displsRow, size);
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


		printf("\n");
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
	    
	    //invio dati di testing
	    MPI_Bcast(testingData, P*M, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
	    MPI_Bcast(classesTesting, P, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

	    //dati locali che il processo root deve gestire
		float * localTrainingData = (float *) malloc(countsRow[rank] * sizeof(float));
		uint8_t * localClassesTraining = (uint8_t*)  malloc(countsRow[rank]/M *sizeof(uint8_t));	

		if(localTrainingData == NULL || localClassesTraining == NULL){
			printf("Not enough memory!\n");
			exit(EXIT_FAILURE);
		}
	    
		//Invio ad ogni processo la porzione di dati da processare
   		MPI_Scatterv(trainingData, countsRow, displsRow, MPI_FLOAT, localTrainingData, countsRow[root_rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
   		
   		// e le relative label
   		MPI_Scatterv(classesTraining, countsClasses, displsClasses, MPI_UINT8_T, localClassesTraining, countsClasses[root_rank], MPI_INT, root_rank, MPI_COMM_WORLD);
   		
		//risultato delle classificazioni
   		int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);

   		int* countsLabel = (int*) malloc(sizeof(int)*LABELS);

   		//tutte le label
   		uint8_t* all_K_label = (uint8_t*) malloc(sizeof(uint8_t)* size * P * K);

   		float*  all_K_distance = (float*) malloc (sizeof (float)* size * P *K);

   		uint8_t *local_k_label = (uint8_t*) malloc(sizeof(uint8_t)* P * K);
		float* local_k_distance = (float*) malloc(sizeof(float)* P * K);

   		if(confusionMatrix == NULL || local_k_label == NULL || local_k_distance == NULL || all_K_label == NULL || all_K_distance == NULL){
			printf("Not enough memory!\n");
			exit(EXIT_FAILURE);
		}


		
			

		calculatelocalKNN(testingData, localTrainingData, localClassesTraining, countsRow[rank]/M, local_k_label, local_k_distance);
		
		
		MPI_Gather(local_k_label, P*K, MPI_UINT8_T, all_K_label, P*K, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

		MPI_Gather(local_k_distance, P*K, MPI_FLOAT, all_K_distance, P*K, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
		
		/*
		printf("Stampo il ricevuto\n");
		for(int j=0; j<size; j++ ){
			for(int i=0; i<P; i++){
				for(int k=0; k<K; k++){
						printf("Distanza %f label %d ",all_K_distance[j*P*K +i*K +k], all_K_label[j*P*K +i*K +k]);
				}
				printf("\n");
			}
			printf("\n\n");
		}
		printf("\n\n\n");
		*/
		
		//ora bisogna aggiornare man mano i dati e calcolare la matrice di confusione
		//uso i dati locali come migliori e se trovo dati migliori aggiorno
		//iesmi matrice del processo 
		for(int i=1; i< size; i++){
			//per ogni riga
			for(int j=0; j< P; j++){
				//se il valore peggiore di distanza corrente è più grande del migliore delle nuove distanze allora aggiorno
				//printf("peggiore distaza corrente %f  migliore nuova distanza %f\n", local_k_distance[j*K + K-1], all_K_distance[i*P*K + j*K]);
				if(local_k_distance[j*K + K-1] > all_K_distance[i*P*K + j*K]){
					//aggiornamento 
					sortLine(&local_k_distance[j*K], &all_K_distance[i*P*K + j*K], &local_k_label[j*K], &all_K_label[i*P*K + j*K]);
				}
				//for(int k=0; k<K; k++){
				//	printf(" %f label %d ",local_k_distance[j*K +k], local_k_label[j*K +k]);
				//}
				//printf("\n");
				
				
			}
		}
		/*
		printf("Prima di calcolo matrice confusione\n");
		for(int i=0; i<P; i++){
			for(int k=0; k<K; k++){
					printf("Distanza %f label %d ",local_k_distance[i*K +k], local_k_label[i*K +k]);
			
			}
			printf("\n");
		}
		*/



		int errors = 0;
		initilizeArray(confusionMatrix, LABELS*LABELS, 0);
		
		//calcolo matrice di confusione
		for(int i=0; i < P; i++){
			//inizializza a zero il vettore
	        initilizeArray(countsLabel, LABELS, 0);
	        int bestLabel = 0;
	        
	        //per i primi k vicini
	        for(int j=0; j<K; j++){ 
	            //int indice = k_labels[j];
	            int label = local_k_label[i*K +j]; 
	  
	            countsLabel[label] = countsLabel[label] + 1;
	            if(countsLabel[label] > countsLabel[bestLabel])
	                bestLabel = label;
	        }
	        

	        int realLabel = classesTesting[i];
	        if (realLabel != bestLabel)
	            errors = errors + 1;
	        
	        //update confusion matrix
	        confusionMatrix [realLabel * LABELS + bestLabel] = confusionMatrix [realLabel * LABELS + bestLabel] +1;
	    }
        
        //printf("Matrice finale\n");
		//printConfusionMatrix(confusionMatrix);

		//printf("Errori totali: %d\n", errors);
		//printf("Record corretti: %d accuratezza (%.2f%%)\n\n", P - errors, 100 - ((float) errors / P) * 100);
		
		
		if(checkresult == true){
			checkResultKNN(trainingData, testingData, classesTraining, classesTesting, confusionMatrix);
		}
    
		
		//dealloco memoria
		
		free(trainingData); trainingData = NULL;
	    free(testingData); testingData = NULL;
	    free(classesTraining); classesTraining = NULL;
		free(classesTesting); classesTesting = NULL;
		free(localTrainingData); testingData = NULL;
		free(localClassesTraining); classesTesting = NULL;
		free(confusionMatrix); confusionMatrix = NULL;
		

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
	 		float * testingData = (float *) malloc(P* M * sizeof(float));
	 		uint8_t * classesTesting = (uint8_t*) malloc(P *sizeof(uint8_t));
			
			float * localTrainingData = (float *) malloc(countsRow[rank] * sizeof(float));
			uint8_t * localClassesTraining = (uint8_t*)  malloc(countsClasses[rank] *sizeof(uint8_t));

			if(testingData == NULL || localTrainingData == NULL || localClassesTraining == NULL || classesTesting == NULL){
				printf("Not enough memory!\n");
				exit(EXIT_FAILURE);
			}	
	 		
	 		//Ricevo dati di training
	 		MPI_Bcast(testingData, P*M, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
	    	MPI_Bcast(classesTesting, P, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

	    	//ricezione porzione di dati di testing da trattare
	    	MPI_Scatterv(NULL, NULL, NULL, NULL, localTrainingData, countsRow[rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
   		
			MPI_Scatterv(NULL, NULL, NULL, NULL, localClassesTraining, countsClasses[rank], MPI_UINT8_T, root_rank, MPI_COMM_WORLD);
   			

   			uint8_t *local_k_label = (uint8_t*) malloc(sizeof(uint8_t)* P * K);
			float* local_k_distance = (float*) malloc(sizeof(float)* P * K);

   			if(local_k_label == NULL || local_k_distance == NULL){
				printf("Not enough memory!\n");
				exit(EXIT_FAILURE);
			}
		

			calculatelocalKNN(testingData, localTrainingData, localClassesTraining, countsRow[rank]/M, local_k_label, local_k_distance);
		
			MPI_Gather(local_k_label, P*K, MPI_UINT8_T, NULL, 0, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

			MPI_Gather(local_k_distance, P*K, MPI_FLOAT, NULL, 0, MPI_FLOAT, root_rank, MPI_COMM_WORLD);

			
	 		free(testingData); testingData = NULL;
	 		free(classesTesting); classesTesting = NULL;
		    free(localTrainingData); localTrainingData = NULL;
			free(localClassesTraining); localClassesTraining = NULL;
			
		
	 	}

	//tutti i processi devono chiudere correttamente MPI
 	MPI_Finalize();
	return 0;

}
