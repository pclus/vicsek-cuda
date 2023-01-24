# CUDA Vicsek model simulator

For details on the model and usage of the noise refer to: [Clusella and Pastor-Satorras, Chaos 31, 043116 (2021)](https://doi.org/10.1063/5.0046926). This code is for the Vicsek model defined on Euclidian space only.

## Disclaimer

I am not a CUDA expert.

## External tools

The code makes a fundamental use of [thrust](https://docs.nvidia.com/cuda/thrust/index.html) in two cases: 
- To compute the average polarization of the system (function `compute_phi` in file `system.cu`)
- To sort the box list `devBox` with `thrust::sort_by_key` in `vicsek.cu`.

Also:

- We use [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/doc/html/) to compute moving window statistics of the order paremter `phi`. This is not fundamental and can be easily changed in the `vicsek.cu` file.
- Most output files can be easily plotted using [gnuplot](http://www.gnuplot.info/), but any other plotting software can work provided it can read binary files.

## Compilation

From the *source* folder, the code can be compiled with

```
nvcc system.cu vicsek.cu -O3 -lcuda -lcurand -lgsl -lgslcblas -o ../gpuvs
```
## System parameters and input

Upon compilation the program can be executed from the workspace with

```
./gpuvs <flag> <L> <rho> <eta>
```
where

- *flag* is a string to identify output files
- *L* is the lattice size
- *rho* is the density of particles
- *eta* is the noise strength. Notice that different types of noise are impletented in the code,
the user needs to comment/uncomment them manually in the `update_bitxus()` function at `system.cu`.

Particle number `N` is directly computed from *rho* and *L*.
The box size for easy particle location is fixed to be 1.

A typical call to the program could be

```
./gpuvs_v1 experiment_A 128 2.0 0.62
```

Other system parameters need to be changed directly in the code:

- The simulation time `Tf` and transient `trans` are defined in the `main()` function in `vicsek.cu`. This is, by default, `Tf=2.5e5, trans=5e4`.
- The distance radius for the nearest neighbour computation is the macro `R0` defined in `system.cuh`. However **this should not be increased to more than 1** without changing some important parts of the code. Otherwise one might need to look at more than 8 neighbouring boxes to compute the local polarization, and this is hardcoded.
- The constant particle speed is the macro `V0` defined in `system.cuh`.

## Workspace structure and output files

Any of the following can be easily adapted modifying the source code `vicsek.cu`.

Output files are stored in three different folders, that **need to be created before**:
- *timeseries* stores the binary file containing the time series of the order parameter
- *outputs* stores the final statistics of phi in plain format
- *snapshots* stores the position and velocity of each particle in binary format

The output files are authomatically named with the provided values of L, rho, and eta,
and also with the "flag" string passed to the program.

Using **gnuplot** one can plot the time series of phi with:

```
plot 'timeseries/<filename>.bin' '%double' using 0:1 with lines
```

The snapshots of the system can be also plotted using, for instance,

```
plot 'snapshots/<filename>.bin' binary format='%double%double%double%double' using 1:2:(atan2($4,$3)) with dots lc palette
```

## Algorithm 

The program is based on the usage of **bitxus** in **boxes**, i.e., [cell lists](https://en.wikipedia.org/wiki/Cell_lists).

A **Bitxu** is a `struct` defined in `system.cuh` that contains all the information of each particle of the Vicsek model, namely, its box location, position, velocity, and local polarization.
The word *bitxu* comes from a direct (and wrong) spelling of the spanish word "bicho" (bug) in catalan pronunciation.
If you are confused by the word, you can think that a *bitxu* is a *particle*, *bird*, *fish*, *bacteria*, *sheep*, or whatever your are trying to model.

A **box** is not explicitly defined in the code, and is one of the L^2 squares of unit size in the lattice. This is the *cell* in cell lists.

The Vicsek model is an array of "bitxus", that in our code is called `devNodes`.
Each *bitxu* is at each time in one and only one box. Knowing what box a *bitxu* belongs to is easy to compute, and it is at all times stored in `devNodes[i].idbox`.

However, since *bitxus* move in time, knowing how many and which *bitxus* are in a given box at a specific moment is difficult.
This is the main problem to be solved.
To address it I create a list called `devBox`.
Counterintuitively, this is not a list of boxes, but a list of *bitxus*'.
This list needs to be **sorted at all times**.
Therefore, in a system with 10 *bitxus* and 4 boxes, `devBox` could look like
```
devBox=[ 0 0 1 1 1 2 3 3 3 3]
```
where it is clear that we have 2 particles in box 0, and 3 in box 1, 1 in box 2 and 4 in box 3.

Since this list is **sorted at all times**, in order to access the elements of `devBox` easily we can just define two arrays, called `devIni` and `devEnd`, of size L^2, i.e., the total number of boxes.
These vectors respectively indicate the initial and final index of the particles in a given box of `devBox`.

Therefore, in our previous example,
```
devIni=[0 2 5 6]
```
and 
```
devEnd=[1 4 5 9]
```
since the first *bitxu* of box 0 is referenced at `devBox[0]` and its final *bitxu* is referenced at `devBox[2]`,
whereas for box 2 we only have one subject which is at position 5, and therefore `devIni[2]=devEnd[2]=5`.

When the array `devBox` changes, `devIni` and `devEnd` need to be updated. This is done by the function `get_positions()`, defined in `system.cu`. 
Thanks to this scheme, it is easy to retrieve all the particles in a specific box.
To compute the local polarization
of a particle one just needs to look at the particles in its own box and the surrounding boxes. This is done by the function `compute_distances()`.

At any time steps, bitxus change boxes, and therefore, the array `devBox` needs to be updated.
This is done in two steps. 

1. When particles move, they update their position in `devBox`.
This is done in the function `update_bitxus()` in `system.cu` (where `devBox` is called `boxlist`).
However, now `devBox` is not sorted.
2. We call `sort_by_key` from `thrust` to sort `devBox`. However, now, particles in `devBox` are not properly identified by their index,
we need to re-organize particles in `devNodes` the same way they have been reorganized in `devBox`. This reorganization is given by the array  `devSL`. We then just need to make `devNodes[i]=devNodes[devSL[i]]`.
This is done with the help of an auxiliary array `devTemp` (despite its name, this is not a temporal array at all), and the  copy and swap functions.






