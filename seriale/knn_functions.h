#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS

//calcola della distanza euclidea tra un sample del train e uno del test
float computeDist(float* train, float* test);

// ordina e aggiorna le distanze e gli indici
void sort(float * distance, int * index);

#endif

