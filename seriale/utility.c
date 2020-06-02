#include "utility.h"
#include "input.h" 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>


void read_file(const char *filename, int lines, int Nfeatures, float* data, int * labels) {
  char path[100] = "../dataset/"; 
  strcat( path, filename );
  FILE *file = fopen(path, "r");
if (file == NULL){
	printf("Impossibile leggere il file!");
	exit(EXIT_FAILURE);
}
else{
  for (int i = 0; i < lines; i++) {
    for (int j = 0; j < Nfeatures; j++) {
      fscanf(file, "%f", &data[i * Nfeatures + j]);
    }
    float label;
    fscanf(file, "%f", &label);
    labels[i] = (int) label;
  }
}
}


void initilizeArray(int * array, int size, int value){
	for(int i=0; i< size; i++){
		array[i] = value;
	}
}

// salva il risultato su file 
int saveResultsOnFile(float time){
	FILE *fp;

	int i, j;
	char * wheretoprint = "resultsKNN_serial.out";
	fp = fopen(wheretoprint,"a");

	if (fp == NULL) {
	    printf("\nCannot write on %s\n", wheretoprint);
	    return -1;
	}

	fprintf(fp, "Test with K = %d trainingData %d x %d and testingData: %d x %d, time: %f\n\n",K, N, M, P, M, time);
	  
	fclose(fp);

	return 0;
}


/*void initilizeDistances(struct distAndLabel* distances){
	//per ogni sample del testing
	for(int i =0; i< P; i++){
		//per ogni vicino
		for(int j=0; j< K; j++){
			//assegno il valore massimo rappresentabile
			distances[i* K + j].distance = LONG_MAX;
			distances[i* K + j].label = -1;
			//printf("%f\n", distances[i* K + j].distance);

		}
	}

}
*/

void printData(float * data, int* labels, int size){
	for(int i=0; i< size; i++){
		for(int j=0; j <M; j++)
			printf(" %f ", data[i*M +j]);
		printf("Classe %d\n", labels[i] );
	}
	printf("\n");
}

void printConfusionMatrix(int* confusionMatrix){
	printf ("\tReale X Risultato\n");
	for(int i=0; i <LABELS; i++){
		for(int j=0; j < LABELS; j++)
			printf("%d ", confusionMatrix[i* LABELS + j]);
		printf("\n");
	}
}
