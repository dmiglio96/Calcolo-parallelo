#ifndef CHECK_RESULTS 
#define CHECK_RESULTS
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "input.h"
#include "utility.h"
#include "knn_functions.h"

//check result
void checkResultKNN(float* trainingData, float* testingData,  uint8_t* classesTraining,  uint8_t* classesTesting, int* confusionMatrix);
#endif