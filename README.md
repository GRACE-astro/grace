# GRACE

`GRACE` is an evolution framework that uses Finite Difference
methods to simulate relativistic spacetimes and plasmas.

## QuickStart guide

In this guide I'll give you the basic steps to run the first `GRACE` simulation and analyze its output, more in depth guides about dependencies, the build system, runtime parameters and how to contribute to `GRACE` can be found in other pages of this documentation.

### Building the code 
The build system used for this project is Cmake. `GRACE` depends on some external libraries, which should be provided at build time. These are: MPI, Kokkos, p4est, Catch2 (for testing only), YAML-cpp, spdlog, VTK, HDF5 and ZLIB. Inside `env/` are some example configuration files that set the required environment variables to enable Cmake to find all dependencies. `GRACE` supports 2 and 3 spatial dimensions. The number of spatial dimensions in `GRACE` is a compile-time switch that can be set via the `GRACE_NSPACEDIM` cmake variable (see below for an example). The default is 2 dimensions. Moreover, `GRACE` leverages Kokkos as a shared-memory parallelization layer to achieve a (more or less) backend independent code structure. The available backends are: CUDA device, HIP device, host OpenMP and host serial. The active backend is determined at build time through the `GRACE_ENABLE_{DEV}` variables, where `DEV` can be `CUDA`, `HIP`, `OMP`, or `SERIAL`. Note that the Kokkos library being linked must also have been compiled with this device, this is checked at build time. An example of how to build `GRACE` is:
```
    cmake -Bbuild -S./ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=hipcc -DGRACE_NSPACEDIM=3 -DGRACE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
``` 
Which sets up a `HIP` backend, compiles the code in 3D, and includes all debug flags in the compilation. Please note that when compiling for a specific device backend the compiler used must support building device code. Some other configure-time definitions that can be passed to the `GRACE` build system are: `GRACE_ENABLE_GRMHD=ON/OFF` (default OFF) which decides whether the system of equations to be solved is the GR(M)HD system, `GRACE_ENABLE_PROFILING=ON/OFF` (default off) which controls whether profiling and tracing hooks are built into the code (note that unless profiling is enabled at runtime, the hooks add essentially zero overhead to the code performance).

After this step is complete, the `build` directory will contain a configuration. You can now compile that configuration by 
```
cmake --build build --target grace -j 4
```
You can remove the `--target` option of the command above if you want to build all the unit tests as well. 
Once this step is complete the executable called `grace` will be found inside the `build` directory.

### Running the code: Parameter (or config) file

`GRACE`'s runtime behaviour is controlled by a configuration (or parameter) file. The format of this file is `yaml` making them easy to read and write (even programmatically, you can dump a `Python` dictionary into a `yaml` file) and very versatile. Plenty of examples can be found in the `configs` directory in this repository, and this tutorial will use `grmhd_shocktube.yaml` which sets up a shocktube test (duh!). 
The parameter file is broken down into sections, the first one being `amr`:
``` yaml
amr: {
  npoints_block_x: 16,
  npoints_block_y: 16,
  npoints_block_z: 16,
  xmin: -0.5,
  xmax: 0.5,
  ymin: -0.1,
  ymax: 0.1,
  zmin: -0.1,
  zmax: 0.1,
  n_ghostzones: 2,
  initial_refinement_level: 2,
  refinement_criterion: "simple_threshold",
  refinement_criterion_CTORE: 2.,
  refinement_criterion_CTODE: 1.,
  refinement_criterion_var: "U",
  max_refinement_level: 7,
  regrid_every: -1,
  regrid_at_postinitial: false
}
```
This sets up the grid extents to be x [-0.5,0.5], y and z [-0.1,0.1]. Since we are just doing second order finite volume, 2 ghostzones is enough, and this is the value selected by the `n_ghostzones` parameter. The rest of the parameters sets the grid resolution, and is worth commenting in more detail. 
- __Regridding__ is disabled, as can be seen from `regrid_every` which is set to -1. If we set a positive value, that would indicate how often in terms of iterations, or timesteps (same thing!) the grid should be refined and coarsened. The mesh steering criterion is selected via `refinement_criterion`, which is essentially a way to compute a certain error estimate on each grid quadrant on a variable, controlled by `refinement_criterion_var`. The `refinement_criterion_CTORE` is the error threshold above which a quadrant is marked for refinement, and `refinement_criterion_CTODE` is the threshold below which a quadrant is flagged for coarsening. Note that not all flagged quads will be coarsened/refined, since the grid always needs to be 2:1 balanced across quad faces, edges and corners. The `max_refinement_level` parameter is self explanatory. 
- __Resolution__ is not always trivial to compute in `GRACE`, since the meshing algorithm is quite complex. Let's break it down. The grid is represented by a _forest of oct-trees_. Each oct-tree consists of a root node (the level 0 grid, a cube) which can be recursively refined. Initially, the grid is always set up at a uniform resolution, which can be refined before initiating the actual simulation by setting the parameter `regrid_at_postinitial`. The initial uniform resolution is controlled by the `initial_refinement_level` parameter, which control the number of times all trees are recursively refined. The number of quadrants in a tree at a given uniform level is given by (2^D)^l where D is the number of spatial dimensions and l is the refinement level. Each quadrant then comprises a number of cells controlled by `npoints_block_x/y/z`. As an example, if the grid is a cube with x,y,z in [0,1] and `initial_refinement_level` is set to 2, then in 3 spatial dimensions there will be 64 quadrants. If then `npoints_block_x/y/z` are all set to 16, then each quadrant will consist of 16 cells per direction and the resolution will be 1/4/16=0.015625. This is covers the case where the grid is a cube in coordinate space. If one of the coordinates has a larger extent, say that y spans [0,2] in our example, then the grid will consist of two oct-trees stacked in the y-direction, each of which has an extent of [0,1] in each direction. This means that the quadrant (and cell) count doubles, ensuring that the resolution is the same in all directions. So to compute the resolution it's enough to consider a grid which is a cube extending to the smallest of the coordinate extents of the grid along x y and z. 
Confused yet? Me too. Thankfully after installing the companion repository `GRACEpy` you will have access to a command line utility that can compute this for you. Just use `grace_info --grid_info --parfile <parfile>`.

Another important section of a parameter file is the one regarding `IO`:
``` yaml
IO: {
  volume_output_cell_variables: ["rho","press","eps","vel[0]", "zvec[0]"],
  volume_output_base_directory: "output_volume",
  volume_output_every: 230,
  plane_surface_output_cell_variables: [],
  plane_surface_output_every: -1,
  sphere_surface_output_every: -1,
  info_output_every: 20,
  info_output_max_reductions: ["rho", "press", "eps"],
  info_output_min_reductions: ["rho", "press", "eps"],
  info_output_norm2_reductions: [],
  scalar_output_every: 50,
  scalar_output_minmax: ["rho","press","eps"],
  scalar_output_norm2: [],
  scalar_output_integral: [],
  output_use_hdf5: true, 
  hdf5_compression_level: 6,
  hdf5_chunk_size: 8192
}
```
Most of these parameters are self-explanatory. All Output frequencies are expressed in terms of iterations (i.e. timesteps) and -1 means off. All `variables` entries can be passed as a list of strings. Passing the first element of a vector or tensor (i.e. `gamma[0,0]`) as requested volume output will result in the full vector / tensor to be included in the output. Scalar reductions can be requested as min, max, norm2 or integral, and similarly for `info` output (which is just what gets printed on console by `GRACE`). For `hdf5` output, compression level and chunk size (in bytes) can be specified. The default values are usually good enough, but if performance or space are critical some experimentation with these parameters can go a long way.

The `system` section of the parameter file control some runtime behaviour of the code: 
``` yaml
system: {
  console_log_level: "info",
  file_log_level: "trace",
  flush_logs_based_on_severity: true,
  flush_level: "trace"
}
```
These parameters essentially control the verbosity of `GRACE` on the consol and in the logfiles. Logs are written in the `logs` directory which will also contain error files and backtraces should there be any. The priority of messages are in the following order (from least to most important): `trace`,`debug`,`info`,`warn`,`error`,`critical`. A typical running setup has `info` output on console and `debug` or `trace` in logfiles, which provides a reasonable amount of information without being too verbose. Note that the logging system in `GRACE` uses a ring buffer which commits messages to files and console periodically to improve performance. This can mean that messages are lost in the queue if an abnormal termination (a crash) happens. To alleviate this issue, since output can be a critical debugging tool to diagnose a crash, the `flush_logs_based_on_severity` can be set to true. This makes it so that when a message of a given priority or higher is issued, all messages in the buffer are committed to file and the buffer is flushed. You can pair this with `flush_level` which decides what priority forces the flush. 

The `evolution` section of the config file controls the timestepper, how to select the timestep, and when the simulation is finished:
``` yaml
evolution: {
  time_stepper: "rk2",
  cfl_factor: 1.0,
  final_time: 0.4,
  #timestep: 0.00025,
  #timestep_selection_mode: "manual"
}
```

The rest of the parameter file will be specific to the system of equations `GRACE` is compiled to solve. Assuming you are compiling for `GR(M)HD`, you will have this section:

``` yaml
grmhd: {
  id_type: "shocktube"
}
```
The only real parameter here sets which kind of initial data you want. 

The equation of state is controlled by its own section of the config file:
``` yaml
eos: {
  eos_type: "hybrid",
  cold_eos_type: "piecewise_polytrope",

  ye_atmosphere: 0,
  temp_atmosphere: 0,
  rho_atmosphere: 1.e-14,
  eps_maximum: 1.e+05,
  gamma_th: 1.6666,

  piecewise_polytrope: {
    n_pieces: 1,
    gammas: [2.],
    rhos:   [],
    kappa_0: 0.
  }
}
```

### Running the code: this time for real

Once you have a parameter file, running grace is as simple as: 
``` bash 
$ mpirun -n <num_gpus> ./grace --grace-parfile <path_to_parfile>
```
Note that `GRACE` makes use of CPU threads through OpenMP to parallelize certain Host tasks, even if running on GPU. 
Therefore, depending on the system, it is worth enabling multithreading and selecting a reasonable thread affinity on 
the CPU side. Refer to the machine and MPI documentations for details on how to do that.

### Examples

The `examples` directory in the `GRACE` repository contains a few example workflows of how to go from building to running the code to analyzing the output on a few known machines, it is recommended for all new users to start from there (especially if you don't have previous experience with HPC systems and simulation codes!). 
## Building the docs 
`GRACE`'s documentation is automatically generated by Doxygen. To build the docs, make sure you have Doxygen installed on your system, then checkout the git submodule providing the `css` stylesheet for `GRACE`'s documentation
```
git submodule update --init 
```
You should then be able to simply build the docs as normal with Doxygen.