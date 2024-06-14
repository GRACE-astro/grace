.. GRACE documentation AMR Parameters file

.. _amr-parameters:

AMR Parameters
**************

This page contains a full reference of all parameters available for the amr module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``npoints_block_x``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`16`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Number of cells per quadrant in first coordinate direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``npoints_block_y``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`16`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Number of cells per quadrant in second coordinate direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``npoints_block_z``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`16`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Number of cells per quadrant in third coordinate direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``n_ghostzones``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`4`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Number of ghost cells per quadrant in each direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``xmin``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`-1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: minimum value of X coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``xmax``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: maximum value of X coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``ymin``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`-1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: minimum value of Y coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``ymax``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: maximum value of Y coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``zmin``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`-1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: minimum value of Z coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``zmax``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`1.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For cartesian coordinates: maximum value of Z coordinate.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``periodic_x``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`For cartesian coordinates: is the grid periodic in X direction?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``periodic_y``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`For cartesian coordinates: is the grid periodic in Y direction?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``periodic_z``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`For cartesian coordinates: is the grid periodic in Z direction?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``reflection_symmetries``
   * - **Type:**
     - :data-type:`array of bool`
   * - **Default Value:**
     - :default-value:`false, false, false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`For cartesian coordinates: does the grid have reflection symmetries?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``inner_region_side``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`1.0`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For spherical coordinates: length of inner Cartesian region's edge.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``inner_region_radius``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`2.0`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For spherical coordinates: radius of interpolated Cartesian to Spherical region.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``outer_region_radius``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`10.0`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For spherical coordinates: radius of outer spherical region.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``use_logarithmic_radial_zone``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`For spherical coordinates: if true, radial coordinate is logarithmic in the outer spherical region.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``uniform_initial_refinement``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`true`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`If true, the grid is initially set up at a uniform refinement level.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``initial_refinement_level``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`2`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Initial (uniform) refinement level.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``regrid_every``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`100`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Regridding frequency expressed in iterations.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``max_refinement_level``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Maximum allowed refinement level.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``regrid_at_postinitial``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`true`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`Perform regridding after setting initial data?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``postinitial_regrid_depth``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`1`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`How many times to regrid after setting initial data.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`FLASH_second_deriv`
   * - **Allowed Range:**
     - :allowed-range:`FLASH_second_deriv, simple_threshold`
   * - **Description:**
     - :param-description:`Criterion to decide which quadrants to refine/coarsen.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion_reduction``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`max`
   * - **Allowed Range:**
     - :allowed-range:`max, min`
   * - **Description:**
     - :param-description:`How to reduce the criterion over cells in a quadrant.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion_CTORE``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.8`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Minimum value of reduced criterion over a quadrant to refine.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion_CTODE``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.2`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Maximum value of reduced criterion over a quadrant to coarsen.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion_var``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`dens`
   * - **Allowed Range:**
     - :allowed-range:`Any existing evolved or auxiliary variable.`
   * - **Description:**
     - :param-description:`Variable where the criterion should be evaluated.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``refinement_criterion_var_is_aux``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`Is the refinement_criterion_var auxiliary?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``FLASH_criterion_eps``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.01`
   * - **Allowed Range:**
     - :allowed-range:`0+:*`
   * - **Description:**
     - :param-description:`Epsilon parameter for FLASH second derivative refinement criterion.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``prolongation_limiter_type``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`minmod`
   * - **Allowed Range:**
     - :allowed-range:`minmod, monotonized-central`
   * - **Description:**
     - :param-description:`Limiter to be used for prolongation on refined cells (typically minmod unless you're sure you know why not).`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the amr module to their default value.

.. code-block:: yaml

   amr: {
     # Physical extent of the coordinates.
     # If the code is compiled for 2 spatial dimensions, z-parameters
     # are ignored. 
     # If spherical/cylindrical coordinates are selected, xmin and ymin 
     # will be ignored, and in the case of spherical coordinates zmin 
     # will also be ignored.
     xmin: -2.,
     xmax: 2.,
     ymin: -1.,
     ymax: 1.,
     zmin: -1.,
     zmax: 1., 
   
     periodic_x: false,
     periodic_y: false,
     periodic_z: false,
     
     # Reflection symmetries, default is to disable all
     reflection_symmetries: {
       x: false,
       y: false,
       z: false
     } , 
     # Should the grid be uniform at t=0? 
     uniform_initial_refinement: true ,
     # Initial level of refinement (for uniform only!)
     initial_refinement_level: 2,
     # For spherical coordinates (and cylindrical in 3D) only!
     # Near the center, to avoid coordinate singularities,
     # a patch of cartesian
     inner_region_side: 1.,
     inner_region_radius: 2.,
     outer_region_radius: 10., 
     use_logarithmic_radial_zone: false,
   
     npoints_block_x: 16,
     npoints_block_y: 16,
     npoints_block_z: 16,
   
     n_ghostzones: 4,
   
     regrid_every: 100, 
     regrid_at_postinitial: true,
     postinitial_regrid_depth: 1,
     max_refinement_level: 11,
   
     refinement_criterion: "FLASH_second_deriv",
     refinement_criterion_reduction: "max", 
     refinement_criterion_CTORE: 0.8,
     refinement_criterion_CTODE: 0.2,
     refinement_criterion_var:   "dens",
     refinement_criterion_var_is_aux: false,
     FLASH_criterion_eps     : 0.01,
     restriction_interpolator_type: "linear",
     prolongation_limiter_type: "minmod",
     prolongation_interpolator_type: "linear"
   }


