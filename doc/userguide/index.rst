.. _grace-userguide:

User Guide
============

Welcome to the GRACE user guide! Below is a description of how to run GRACE for a variety of configurations, as well 
as of how to analyze the output it produces. 
If you're looking for a guide on how to build the code, please see the :doc:`related page <../code_building_guide/index>`.
If you're wondering what grace even is, please refer to the :doc:`Introduction <../introduction/index>`.


Input parameters
******************

Input parameters control the runtime behaviour of GRACE. They can be provided through a file, aptly called parameter file, which has 
to be written in `YAML <https://en.wikipedia.org/wiki/YAML>`__ format. The parameter file can be passed to the GRACE executable with 
the following invocation:

.. code-block:: bash

    $ mpirun -n <num_procs> ./grace --grace-parfile ./<parfile_name>.yaml

GRACE supports many different parameters that allow for very detailed customization of the runtime behaviour
of the code. We will now give a list of all possible parameters, their type, their default value and the range 
of values they support. Parameters in the ``yaml`` config file are split into sections, which roughly correspond 
with GRACE's modules, so we'll provide description of the parameters together grouped by the module they belong to.

.. toctree::
   :maxdepth: 1
   :caption: Code Modules

   params-m1
   params-gw_integrals
   params-bh_diagnostics
   params-coordinate_system
   params-checkpoints
   params-IO
   params-spherical_surfaces
   params-puncture_tracker
   params-grmhd
   params-evolution
   params-system
   params-z4c
   params-amr


Output Data Formats 
*********************

There are a few different kinds of output in GRACE. 

First, there is scalar output. This encompasses reductions of any registered variable (auxiliary or evolved) in GRACE.
Possible reductions are: coordinate volume integral, L2 norm, max and min. The output frequency of all scalar quantities is controlled by a single parameter, 
and all output can be found in the same directory. Output files contain three columns: iteration, simulation time, and the value. 

The second kind of output is also in the form of a timeseries and comprises all spherical surface integrals. These are implemented 
as custom modules in GRACE which have their own parameters. One example is the ``gw_integrals`` module, which simply takes the names 
of registered surfaces where integrals should be performed, and outputs the spherical harmonic decomposition of the Penrose-Newman scalar 
onto these surfaces. The output frequency here is controlled by the diagnostic output frequency, and the corresponding files are placed in 
the same directory as scalars. 

Other outputs are performed in HDF5 by default, with the legacy (deprecated) option of native VTK. GRACE supports volume and plane surface output (in xy, xz, and yz planes), as 
well as output of point data on spherical surfaces. All this data can be easily visualized in `ParaView <https://www.paraview.org/>`_ through the use of XDMF descriptors (see the :doc:`related page <../python_interface/index>` on how to generate them), 
as well as in python through the vtk Python interface. 

Code Description
*********************

We now provide a minimal description of the most relevant aspects of the code from a user's perspective (I think!). 

Variable Names and Definitions 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRACE evolves the equations of GRMHD on static or dynamical curved backgrounds. 

In particular, the set of conserved variables in GRACE corresponds with the standard Valencia 3+1 formulation of GRMHD. 
Inside the code, the conserved rest-mass density is called ``dens``, the conserved energy is called ``tau``, and the 
conserved momentum density ``stilde[0], stilde[1], stilde[2]``. 
As hydrodynamic primitives we adopt the cell centered magnetic field ``Bvec[0], Bvec[1], Bvec[2]``, the rest mass density ``rho``,
the pressure ``press``, the temperature ``temperature``, the specific internal energy ``eps``, the entropy per particle ``entropy``, 
the electron fraction ``ye``, and the velocity, which is stored as :math:`z^i = W v^i` where :math:`W` is the Lorentz factor and 
:math:`v^i` is the standard Eulerian velocity. The velocity in GRACE is called ``zvec[0] zvec[1] zvec[2]``. Additionally, we store the 
divergence of the magnetic field ``Bdiv`` as well as a proxy measure for the conservative to primitive inaccuracy, ``c2p_err``.
The MHD equations are written in Heaviside-Lorentz units. 

When the metric is Cowling (not evolved), it is stored as physical spatial metric ``gamma[0,0] gamma[0,1] gamma[0,2] gamma[1,1] gamma[1,2] gamma[2,2]``, 
physical extrinsic curvature ``ext_curv[i,j]``, shift vector ``beta[0] beta[1] beta[2]`` and lapse function ``alp``. 

When the metric is evolved (Z4c) only the conformal metric is stored, and we have ``gamma_tilde[i,j]`` (metric), ``A_tilde[i,j]`` (conformal traceless extrinsic curvature)
``conf_fact`` (conformal factor), ``z4c_Khat`` (trace of extrinsic curvature minus constraint), ``z4c_theta`` (the constraint propagating dof), ``z4c_Gamma[i]`` (Shibata-Nakamura like Gamma tilde),
``z4c_Bdriver`` (hyperbolic gamma driver auxiliary), ``alp`` and ``beta[i]``.
Moreover we have the additional auxiliary variables ``z4c_H`` and ``z4c_M[i]``, representing the constraint violations, and ``PsiRe``, ``PsiIm`` representing the Penrose-Newman scalar. 
Note that the ``Psi`` is not updated after each step like other auxiliary variables, being instead computed whenever diagnostic quantities are computed. 


Grid Setup
~~~~~~~~~~~~~~~

The grid in GRACE is represented as a forest of oct-trees. If you're unfamiliar with the concept please check out the great material that can be 
found online on the topic. Essentially, this means that the grid is represented by a series of root nodes, each of which can be successively refined 
into sets of eight children. These refined nodes are a series of oct-trees, and the leaves of this tree (the deepest node in each branch that has no direct descendants)
form the actual grid. These leaves can be arranged in a one-dimensional sequence by so-called morton or z-ordering. In GRACE these trees is a cube in carteisan coordinates, and each of its leaves is called quadrant,
following the nomenclature of ``p4est`` (technically they are called ``octants`` in ``p8est``, the 3D version of the library). Each quadrant itself consists of a set of cells in each spatial direction, which is 
extended by a set of ghostzones on either side. This means that each quadrant is a grid block consisting of equally spaced cells + ghost-zones, where 
differential operators can be self-consistently evaluated.

Operationally, the grid is split into trees according to its sizes along each axis. One tree will have sides equal to the smallest grid extents, and equal trees will be stacked 
to set up the user requested grid. For example, if the grid spans :max:`(x,y,z)\in [-1024,1024]\times [-1024,1024] \times [0,1024]` each tree will be :max:`1024\times 1024 \times 1024` and 
two trees will be stacked in the x and y directions. 
After the trees are set up, the grid is initialized at a uniform refinement level :max:`l`, controlled by the parameter ``amr::initial_refinement_level``, so that each of the trees will consist of :math:`2^l` quadrants 
per direction. Continuing the previous example, if the uniform refinement level is set to one each quadrant's side would be :max:`512` long. 
Finally, assuming ``amr::npoints_block_x = 32`` the initial grid from the example would have a base resolution of ``1024/2/32``. 

On top of this uniform grid GRACE allows the user to set up an arbitrary amount of fixed mesh refinement (FMR) boxes, specified by coordinate extents. Each of these halves the grid spacing. 
Finally, the grid can be adapted during the simulation with AMR. In this case, the base grid set up as uniform + FMR is untouched by the AMR, which can optionally refine on top of that and remove 
these additional refined quadrants. 

AMR in GRACE is controlled by the regrid frequency ``amr::regrid_every`` (expressed in iterations), as well as by the criterion to be used. This criterion provides an error estimate, 
which is reduced to one number per quadrant :math:`\epsilon(q)`. Refinement is triggered if this error exceeds a user defined threshold ``amr::refinement_criterion_CTORE`` or coarsened if 
the error is lowered than another threshold ``refinement_criterion_CTODE``. 
When using AMR it can be useful to set up a minimal FMR grid and let the AMR refine on top of that already in the initial data. This can be done by using ``amr::regrid_at_postinitial`` and 
``amr::postinitial_regrid_depth``.
