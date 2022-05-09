/* Author: Pau Clusella */

#ifndef STD_H
#define STD_H
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <time.h>
#endif

// CUDA and thrust libraries
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

// GNU Scientific Library, only used for rstat
#ifndef GSL_H
#define GSL_H
        #include <gsl/gsl_math.h>
        #include <gsl/gsl_rng.h>
        #include <gsl/gsl_rstat.h>
        #include <gsl/gsl_randist.h>
#endif

#define INDEX(i,j) (MOD(i,L)*L+MOD(j,L))
#define MOD(x,L) (( (x) % L + L) %L)

#define NORMA(x,y) (sqrt((x)*(x)+(y)*(y)))
#define N_ORMA(x,y) ((x)*(x)+(y)*(y)) // it is easy to compute a square rather than a root

#define R0 (1.0) // Distance radius
#define V0 (0.5) // Particle speed

// Each particle in the Vicsek model is a structure called "Bitxu"
typedef struct node{
	int idbox;	// box index where the particle is located
	double px;	// position x
	double py;	// position y
	double vx;	// velocity x
	double vy;	// velocity y
	double bx;	// local polarization x
	double by;	// local polarization y
}Bitxu;

__global__ void setup_kernel(curandState *state, int seed);
__global__ void setup_list( int *sorted_indices , int N);
__global__ void setup_neigh( int *neighbour, int L );
__global__ void get_positions( int *boxlist, int *init, int *end , int N);
__global__ void compute_distances( Bitxu *nodes, int *neigh, int *init, int *end);

__global__ void initial_conditions(Bitxu *nodes, int *boxlist, curandState *state, int L);
__global__ void update_bitxus(Bitxu *nodes, int *boxlist, int *sorted_indices, curandState *state , double eta, int L);

__global__ void retreive_velocities( Bitxu *devNodes, double *boxvel_x, double *boxvel_y);
double compute_phi( double *boxvel_x, double *boxvel_y, int N);

__global__ void copy_nodes( Bitxu *a, Bitxu *b  );
__global__ void swap_nodes( Bitxu *a, Bitxu *b, int *sorted_indices  );
