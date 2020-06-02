#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H 

void read_file(const char *filename, int lines, int Nfeatures, float* data, int * labels);

void initilizeArray(int * array, int size, int value);

// salva il risultato su file 
int saveResultsOnFile(float time);

//void initilizeDistances(struct distAndLabel* distances);

void printData(float * data, int* labels, int size);

void printConfusionMatrix(int* confusionMatrix);

#endif