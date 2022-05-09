/* Author: Pau Clusella */

#ifndef SYSTEM_H
#define SYSTEM_H
        #include "system.cuh"
#endif


__global__ void retreive_velocities( Bitxu *nodes, double *boxvel_x, double *boxvel_y)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	Bitxu *bitxu=nodes+id;
	boxvel_x[id]=bitxu->vx;
	boxvel_y[id]=bitxu->vy;
	return ;
}

// Global polarization using thrust reduction
double compute_phi(double *boxvel_x, double *boxvel_y, int N)
{
	thrust::device_ptr<double> X(boxvel_x);
	thrust::device_ptr<double> Y(boxvel_y);
	thrust::device_vector< double > vX(X, X+N);
	thrust::device_vector< double > vY(Y, Y+N);
        double x = thrust::reduce(vX.begin(), vX.end());
        double y = thrust::reduce(vY.begin(), vY.end());
	return sqrt(x*x+y*y)/(1.0*N);
}

// Initialization of random number generator
__global__ void setup_kernel(curandState *state, int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init( seed , id, 0, &state[id]);
	return ;
}

// Need to setup a list before sorting all the particles, so we know their ids
__global__ void setup_list( int *sorted_indices , int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	sorted_indices[id]=id;
	return ;
}

// Each box has 8 neighbouring boxes, the indices are stored from the beginning
__global__ void setup_neigh( int *neighbour, int L )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int i=id/L;
	int j=id%L;
	int jj=8*id;
        {
                neighbour[jj++]=INDEX(i+1,j);
                neighbour[jj++]=INDEX(i-1,j);
                neighbour[jj++]=INDEX(i,j+1);
                neighbour[jj++]=INDEX(i,j-1);
                neighbour[jj++]=INDEX(i+1,j+1);
                neighbour[jj++]=INDEX(i+1,j-1);
                neighbour[jj++]=INDEX(i-1,j+1);
                neighbour[jj++]=INDEX(i-1,j-1);
        }

	return ;
}

// Computed the initial and final position of the particles in each box in the total array
__global__ void get_positions( int *boxlist, int *init, int *end , int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = boxlist[i];
	int test1=( i==0 ? 1 :  k - boxlist[i-1] );
	int test2=( i==N-1 ? 1 :  boxlist[i+1]-k );
	if( test1>0 )
	{
		init[ k ] = i; 
	}
	if( test2>0 )
	{
		end[ k ] = i; 
	}
	return ;
}

// Actual computation of distances between particles
__global__ void compute_distances( Bitxu *nodes, int *neigh, int *init, int *end) 
{
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	Bitxu *bitxu=nodes+i;
	Bitxu *next;
	int idbox=bitxu->idbox;
	int kk=0;
	double bx=0;
	double by=0;
	double px=bitxu->px;
	double py=bitxu->py;
	int jj,i0,in;

	
	for( int j=0 ; j<9 ; j++ )	// For each neighbouring box (and its own box)...
	{
		jj = ( j==8 ? idbox : neigh[ 8*idbox + j ] );	
		i0 = init[jj];
		in = end[jj];
		for(int k=i0 ; k<=in ; k++ )	// ...compute the distance between their particles
		{
			next = nodes + k;
			double px0=next->px;
			double py0=next->py;
			if( N_ORMA( (px-px0) , (py-py0) ) <= R0 )
			{
				bx += next->vx;
				by += next->vy;
				kk++;
			}
		}
	}
	bitxu->bx = ( kk==0 ? 0.0 : bx / (1.0*kk) );
	bitxu->by = ( kk==0 ? 0.0 : by / (1.0*kk) );

	return ;
}

// System initialization
__global__ void  initial_conditions(Bitxu *nodes, int *boxlist, curandState *state, int L)
{
        int idbox;
        double theta;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
        Bitxu *bitxu=nodes+i;
        //theta=curand_uniform_double(&state[i])*2*M_PI;		// Disordered
        theta=0;                              		// Ordered
        bitxu->px=curand_uniform_double(&state[i])*L;
        bitxu->py=curand_uniform_double(&state[i])*L;
        bitxu->vx=cos(theta);
        bitxu->vy=sin(theta);
        idbox=(int)bitxu->px+L*(int)bitxu->py;
	bitxu->idbox=idbox;
	boxlist[i]=idbox;

        return ;
}


__global__ void update_bitxus(Bitxu *nodes, int *boxlist, int *sorted_indices, curandState *state , double eta, int L)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	curandState localState = state[i];
	Bitxu *bitxu=nodes+i;
	double bx=bitxu->bx;
	double by=bitxu->by;
	double px=bitxu->px;
	double py=bitxu->py;
	double norm,x,y;
	
	//--------------------------------------
	/* Bivariate Normal noise */
	//double2 r = curand_normal2_double(&localState);
	//x=bx+eta*r.x;
	//y=by+eta*r.y;
	//norm= sqrt ( x*x + y*y );
	//x=x/norm;
	//y=y/norm;
	//--------------------------------------

        //--------------------------------------
	/* Vectorial noise */
	double theta=M_PI*(1.0 - 2.0*curand_uniform_double(&localState));
	x=bx+eta*cos(theta);
	y=by+eta*sin(theta);
	norm=sqrt( x*x + y*y );
	x=x/norm;
	y=y/norm;
	//--------------------------------------
	
	//--------------------------------------
	/* Wrapped Normal noise */
	//norm=sqrt( bx*bx + by*by );
	//bx=bx/norm;
	//by=by/norm;
	//double theta= (eta/norm)*curand_normal_double(&localState);
	//double cc=cos(theta);
	//double ss=sin(theta);
	//x=bx*cc-by*ss;
	//y=bx*ss+by*cc;
	//--------------------------------------

	
	// Update box and position
	px=px + V0*x;
	py=py + V0*y;
	px=(px > L ? px-L : px );
	px=(px < 0 ? px+L : px );
	py=(py > L ? py-L : py );
	py=(py < 0 ? py+L : py );
	
	int id=(int)px + L*((int)py);
	bitxu->idbox=id;
	bitxu->px=px;
	bitxu->py=py;
	bitxu->vx=x;
	bitxu->vy=y;
	state[i]=localState;
	boxlist[i]=id;
	sorted_indices[i]=i;

	return ;
}

// Copies particle in a[i] into b[i]
__global__ void copy_nodes( Bitxu *a, Bitxu *b  )
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	Bitxu *x=a+i;
	Bitxu *y=b+i;
	y->idbox=x->idbox;
  	y->px=x->px;
  	y->py=x->py;
  	y->vx=x->vx;
  	y->vy=x->vy;
	return ;
}

// Copies particle b[sorted_indices[i]] to a[i]
__global__ void swap_nodes( Bitxu *a, Bitxu *b, int *sorted_indices  )
{
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        Bitxu *x=b+sorted_indices[i];
        Bitxu *y=a+i;
        y->idbox=x->idbox;
        y->px=x->px;
        y->py=x->py;
        y->vx=x->vx;
        y->vy=x->vy;
        return ;
}

