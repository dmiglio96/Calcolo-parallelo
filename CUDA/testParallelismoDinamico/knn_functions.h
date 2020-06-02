#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>
__device__ float distanceFunction(float* train, float* test);


__global__ void computeDist_kernel(const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ dev_distances);//, int* dev_labels);

__global__ void sort_kernel(float* __restrict__ dev_distances, int* __restrict__ dev_labels);
#endif

