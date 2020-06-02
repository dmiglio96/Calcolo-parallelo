#ifndef CHECK_RESULTS 
#define CHECK_RESULTS
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "input.h"
#include "utility.h"
#include "knn_functions.h"

//calcola della distanza euclidea tra un sample del train e uno del test
float computeDist(float* train, float* test);

// ordina e aggiorna le distanze e gli indici
void sort(float * distance, int * index);

//check result
void checkResultKNN(float* trainingData, float* testingData, int* classesTraining, int* classesTesting, int* confusionMatrix);
#endif
