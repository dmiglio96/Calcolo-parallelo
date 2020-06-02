#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>
//calcola della distanza euclidea tra un sample del train e uno del test
float computeDist(float* train, float* test);

// ordina e aggiorna le distanze e gli indici
void sort(float * distance, int * index);

void suddivisioneTestingData(int* countsRow, int* displsRow, int size);

void suddivisioneClasses(int* countsClasses, int* displsRow, int size);

int calculatelocalKNN(float* trainingData, float* localTestingData, uint8_t* classesTraining, uint8_t* localClassesTesting, int NrecordTesting, int * localResultCM);
#endif



