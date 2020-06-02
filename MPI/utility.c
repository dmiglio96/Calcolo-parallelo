#include <stdint.h> 
#include "utility.h"
#include "input.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>


void read_file(const char *filename, int lines, int Nfeatures, float* data, uint8_t * labels) {
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
      //printf("i %d j %d data[i*line +j] %f\n",i, j, data[i * Nfeatures + j]);
    }
    float label;
    fscanf(file, "%f", &label);
    labels[i] = (uint8_t )label;
  }
}
}


void initilizeArray(int * array, int size, int value){
	for(int i=0; i< size; i++){
		array[i] = value;
	}
}

// salva il risultato su file 
int saveResultsOnFile(float time, int size){
	FILE *fp;

	int i, j;
	char * wheretoprint = "resultsKNN_mpi.out";
	fp = fopen(wheretoprint,"a");

	if (fp == NULL) {
	    printf("\nCannot write on %s\n", wheretoprint);
	    return -1;
	}

	fprintf(fp, "Test with %d process K = %d trainingData %d x %d and testingData: %d x %d , time: %f\n\n",size, K, N, M, P, M, time);
	  
	fclose(fp);

	return 0;
}


void printData(float * data, uint8_t* labels, int size){
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
