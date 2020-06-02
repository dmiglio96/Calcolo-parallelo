#ifndef UTILITY_FUNCTIONS_H 
#define UTILITY_FUNCTIONS_H
#include <stdint.h>
void read_file(const char *filename, int lines, int Nfeatures, float* data, uint8_t * labels);

void initilizeArray(int * array, int size, int value);

// salva il risultato su file 
int saveResultsOnFile(float time, int size);

//void initilizeDistances(struct distAndLabel* distances);

void printData(float * data, uint8_t* labels, int size);

void printConfusionMatrix(int* confusionMatrix);

#endif