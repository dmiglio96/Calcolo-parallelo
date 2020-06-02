#ifndef LBP_FUNCTIONS
#define LBP_FUNCTIONS 
#include <stdint.h>

__global__ void computeDist_kernel(const float *  __restrict__train, size_t pitchTrain, const float*  __restrict__ test, size_t pitchTest, float*  __restrict__ dist, size_t pitchDist);

__global__ void sort_kernel(float*  __restrict__ dev_distances, size_t pitchdist, int*  __restrict__ dev_labels, size_t pitchLabel);
#endif

