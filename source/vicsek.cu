/* Author: Pau Clusella */

#ifndef GPUSYSTEM_H
#define GPUSYSTEM_H
        #include "vicsek.h"
#endif

int main(int argc, char **argv)
{
        //----------------------------------------------------
        int L=atoi(argv[2]);		// lattice length
        double rho=atof(argv[3]);	// particle density
        double eta=atof(argv[4]);	// noise level
        int N=(int)(rho*L*L);		// number of particles

        {
                fprintf(stderr,"\n\033[1;35mL = %d\n", L);
                fprintf(stderr,"rho = %lf\n",rho);
                fprintf(stderr,"N = %d\n",N);
                fprintf(stderr,"eta = %lf\n",eta);
                fprintf(stderr,"v0 = %lf\n",V0);
                fprintf(stderr,"\033[0m\n");
        }
        //----------------------------------------------------

        //--------- RANDOM NUMBER GENERATOR --------------
        curandState *devStates;
        cudaMalloc((void **)&devStates, N*sizeof(curandState));
        setup_kernel<<< (N+255)/256, 256 >>>(devStates, time(0));
        //------------------------------------------------

        //----------------------------------------------------
	double phi;
        Bitxu *devNodes,*devTemp;
	int   *devBox;
	int   *devNeigh;
	double *boxvel_x,*boxvel_y;
	int *devIni, *devEnd, *devSL;

	/* allocate memory in the  GPU */
        cudaMalloc( &devNodes, sizeof(Bitxu)*N );
        cudaMalloc( &devTemp, sizeof(Bitxu)*N );
        cudaMalloc( &devSL, sizeof(int)*N );
        cudaMalloc( &devBox, sizeof(int)*N );
        cudaMalloc( &devEnd, sizeof(int)*L*L );
        cudaMalloc( &devIni, sizeof(int)*L*L );
        cudaMalloc( &devNeigh, sizeof(int)*8*L*L );
        cudaMalloc( &boxvel_x, sizeof(double)*N );
        cudaMalloc( &boxvel_y, sizeof(double)*N );
	
        //------------------------------------------------
	/* System initialization */
        setup_list<<< N_BLOCK , BLOCK_SIZE >>>( devSL , N );
	setup_neigh<<< L_BLOCK , BLOCK_SIZE >>>( devNeigh, L );
        initial_conditions<<< N_BLOCK , BLOCK_SIZE >>>( devNodes , devBox, devStates, L );
	retreive_velocities<<< N_BLOCK , BLOCK_SIZE >>> (devNodes, boxvel_x, boxvel_y);
	phi=compute_phi( boxvel_x, boxvel_y, N);
	fprintf(stderr,"Starting simulation\n");
        //------------------------------------------------

        //------------- READ IC FROM FILE  -------------------------
	// uncomment next two lines in case you want to use initial conditions from previous simulations:
	//sprintf(namm,"init.bin");
	//read_ic(namm,devNodes,devBox,L,N);
        //----------------------------------------------------------

        //----------------- PREPARE GSL STATS -----------
        gsl_rstat_workspace *rstat_p = gsl_rstat_alloc();
        //-----------------------------------------------

        //------------------ SET OUTPUT FILES ------------
        FILE *fout;
        FILE *fout_ts;
        char *namm=(char *)malloc(sizeof(char)*200);
	int snap=0;
        //-----------------------------------------------

	// total and transient computational time:
	double Tf=2.5e5, trans=5e4;

	/* main algorithm*/
	{
		int t=0;
                sprintf(namm,"timeseries/ts_%s_%d_rho%d_eta%.4f.bin",argv[1],L,(int)rho,eta);
        	fout_ts=fopen(namm,"wb");
		gsl_rstat_reset(rstat_p);
        	while( t<Tf ) 
      		{
			if( t%1000==0 ) fprintf(stderr,"Integrating t=%.2f%% \r",t*100/Tf);

			/* sort particles using thrust  */
      			thrust::sort_by_key(thrust::device, devBox , devBox + N, devSL );

			/*  */
			copy_nodes<<< N_BLOCK , BLOCK_SIZE >>>(devNodes , devTemp );	
			swap_nodes<<< N_BLOCK , BLOCK_SIZE >>>(devNodes , devTemp , devSL ); 

			/* initial and final positions of particles in each box */
			get_positions<<< N_BLOCK , BLOCK_SIZE >>>( devBox , devIni , devEnd , N);

			/* compute the local polarization of each particle  */
			compute_distances<<< N_BLOCK , BLOCK_SIZE >>>
				( devNodes, devNeigh , devIni , devEnd );

			/* update particle position using local polarization and noise */
			update_bitxus<<< N_BLOCK, BLOCK_SIZE >>>( devNodes, devBox, devSL, devStates , eta, L);
			t++;
			
			/* compute the order parameter with the help of thrust */ 
		        retreive_velocities<<< N_BLOCK , BLOCK_SIZE >>> (devNodes, boxvel_x, boxvel_y);
        		phi=compute_phi( boxvel_x, boxvel_y, N); 

			/* write time series of the global polarization of the system */
			fwrite(&phi,sizeof(phi),1,fout_ts);
			if( t> trans )
			{
			        gsl_rstat_add( phi, rstat_p);
			}
			if(t%1000==0) fflush(fout_ts);

			/* take some selfies */
			//if(t%100000==0)
			//{
		        //      sprintf(namm,"snapshots/snap_%s_L%d__rho%d_eta%.4f_snap%d.bin",argv[1],L,(int)rho,eta,snap++);
			//	snapshot(namm, devNodes,N);
			//}

        	}
        	fclose(fout_ts);

		//-----------------------------------------------------------
		/* compute and print the statistics of phi using GSL stats */
                sprintf(namm,"outputs/mean_%s_L%d__rho%d_eta%.4f.dat",argv[1],L,(int)rho,eta);
        	fout=fopen(namm,"w");
        	fprintf(stderr,"\n");
        	double mean = gsl_rstat_mean(rstat_p);
        	double var = gsl_rstat_variance(rstat_p);
        	double kurt = gsl_rstat_kurtosis(rstat_p);
        	fprintf(fout,"%lf %.16g %.16g %.16g\n",eta,mean,var,kurt);
        	fclose(fout);
		//-----------------------------------------------------------

                sprintf(namm,"snapshots/snap_%s_L%d__rho%d_eta%.4f.bin",argv[1],L,(int)rho,eta);
		snapshot(namm, devNodes,N);
	}

        free(namm);
        cudaFree(devNodes);
        cudaFree(devTemp);
        cudaFree(devSL);
        cudaFree(devBox);
        cudaFree(devEnd);
        cudaFree(devIni);
        cudaFree(devNeigh);
        cudaFree(boxvel_x);
        cudaFree(boxvel_y);
        return 0;
}

// Print results in a binary file
void snapshot(char *name, Bitxu *devNodes, int N)
{
	FILE *fout_snap;
        fout_snap=fopen(name,"wb");
        Bitxu *nod=(Bitxu *) malloc(sizeof(Bitxu)*N);
        cudaMemcpy( nod, devNodes, N*sizeof(Bitxu), cudaMemcpyDeviceToHost);
        for(int j=0;j<N;j++)
        {
                fwrite(&nod[j].px,sizeof(double),1,fout_snap);
                fwrite(&nod[j].py,sizeof(double),1,fout_snap);
                fwrite(&nod[j].vx,sizeof(double),1,fout_snap);
                fwrite(&nod[j].vy,sizeof(double),1,fout_snap);
        }
        free(nod);
        fclose(fout_snap);
	return ;
}


// Not used at the moment, reads initial conditions from file generated using snapshot()
void read_ic(char *name, Bitxu *devNodes, int *devBox ,int L, int N)
{
        Bitxu *nod=(Bitxu *) malloc(sizeof(Bitxu)*N);
        int *boxlist=(int *) malloc(sizeof(int)*N);
        double a[4];
        FILE *ptr;

        ptr = fopen(name,"rb");  
        fprintf(stderr,"Reading...\n");
        for(int i=0;i<N;i++)
        {
              size_t aux=fread(a,sizeof(double),4,ptr);
              nod[i].px=a[0];
              nod[i].py=a[1];
              nod[i].vx=a[2];
              nod[i].vy=a[3];
              nod[i].idbox=(int)a[0]+L*(int)a[1];
	      boxlist[i]=nod[i].idbox;
        }
        cudaMemcpy( devNodes, nod, N*sizeof(Bitxu), cudaMemcpyHostToDevice);
        cudaMemcpy( devBox, boxlist, N*sizeof(int), cudaMemcpyHostToDevice);
        fclose(ptr);
        free(nod);
        free(boxlist);
        fprintf(stderr,"Reading done\n");
	return ;
}
