/* Author: Pau Clusella */


#ifndef STD_H
#define STD_H
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <time.h>
#endif

#ifndef CUDA_H
#define CUDA_H
        #include <curand.h>
        #include <cuda.h>
        #include <curand_kernel.h>
        #include <thrust/device_vector.h>
        #include <thrust/device_ptr.h>
        #include <thrust/sort.h>
	#include <thrust/execution_policy.h>

        #define BLOCK_SIZE 256
        #define N_BLOCK (N+255)/256
        #define L_BLOCK (L*L+255)/256

#endif

#ifndef SYSTEM_H
#define SYSTEM_H
        #include "system.cuh"
#endif

void snapshot(char *name, Bitxu *devNodes, int N);
void read_ic(char *name, Bitxu *devNodes, int *devBox, int L, int N);
