#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>
//calcola della distanza euclidea tra un sample del train e uno del test
float computeDist(float* train, float* test);

// ordina e aggiorna le distanze e gli indici
void sort(float * distance, uint8_t * index, int numeroElementi);

void sortLine(float * distance, float * distance2, uint8_t* index1, uint8_t* index2);

void suddivisioneTrainingData(int* countsRow, int* displsRow, int size);

void suddivisioneClasses(int* countsClasses, int* displsRow, int size);

void calculatelocalKNN(float* testingData, float* localTrainingData, uint8_t* localClassesTraining , int NrecordTrain, uint8_t *local_k_label, float* local_k_distance);//, int * localResultCM);
#endif


		

