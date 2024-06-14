.. GRACE documentation IO Parameters file

.. _IO-parameters:

IO Parameters
*************

This page contains a full reference of all parameters available for the IO module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``volume_output_every``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`-1:*`
   * - **Description:**
     - :param-description:`Frequency of volume (codimension 0) output in iterations, -1 means never.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``plane_surface_output_every``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`-1:*`
   * - **Description:**
     - :param-description:`Frequency of plane surface (codimension 1) output in iterations, -1 means never.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``sphere_surface_output_every``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`-1:*`
   * - **Description:**
     - :param-description:`Frequency of spherical surface (codimension 1) output in iterations, -1 means never.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_every``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`-1:*`
   * - **Description:**
     - :param-description:`Frequency of scalar output in iterations, -1 means never.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``info_output_every``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`10`
   * - **Allowed Range:**
     - :allowed-range:`-1:*`
   * - **Description:**
     - :param-description:`Frequency of console output in iterations, -1 means never.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``info_output_max_reductions``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary variables.`
   * - **Description:**
     - :param-description:`In the console output, which variables' maxima should be included.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``info_output_min_reductions``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary variables.`
   * - **Description:**
     - :param-description:`In the console output, which variables' minima should be included.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``info_output_norm2_reductions``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary variables.`
   * - **Description:**
     - :param-description:`In the console output, which variables' L2 norms should be included.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``log_output_base_filename``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`grace_log`
   * - **Allowed Range:**
     - :allowed-range:`Any valid filename without extension.`
   * - **Description:**
     - :param-description:`The log-files will be called ` ``<log_output_base_filename>_<Rank>.log`` :param-description:`.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``log_output_base_directory``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`./logs`
   * - **Allowed Range:**
     - :allowed-range:`Any valid relative or absolute path which is user writeable.`
   * - **Description:**
     - :param-description:`The log-files will be placed in this directory.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``volume_output_base_filename``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`volume_out`
   * - **Allowed Range:**
     - :allowed-range:`Any valid filename without extension.`
   * - **Description:**
     - :param-description:`The volume output files will be called ` ``<volume_output_base_filename>_<iteration>.<extension>`` :param-description:`.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``volume_output_base_directory``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`./output_volume`
   * - **Allowed Range:**
     - :allowed-range:`Any valid relative or absolute path which is user writeable.`
   * - **Description:**
     - :param-description:`The volume output files will be placed in this directory.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``surface_output_base_filename``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`surface_out`
   * - **Allowed Range:**
     - :allowed-range:`Any valid filename without extension.`
   * - **Description:**
     - :param-description:`The surface output files will be called ` ``<surface_output_base_filename>_<plane/surface>_<plane/surface_name>_<iteration>.<extension>`` :param-description:`.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``surface_output_base_directory``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`./output_surface`
   * - **Allowed Range:**
     - :allowed-range:`Any valid relative or absolute path which is user writeable.`
   * - **Description:**
     - :param-description:`The surface output files will be placed in this directory.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_base_filename``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`true, false`
   * - **Allowed Range:**
     - :allowed-range:`Any valid filename without extension.`
   * - **Description:**
     - :param-description:`The scalar output files will be called ` ``<scalar_output_base_filename><varname>_<reduction>.<extension>`` :param-description:`.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_base_directory``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`./output_scalar`
   * - **Allowed Range:**
     - :allowed-range:`Any valid relative or absolute path which is user writeable.`
   * - **Description:**
     - :param-description:`The scalar output files will be placed in this directory.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_use_hdf5``
   * - **Type:**
     - :data-type:`boolean`
   * - **Default Value:**
     - :default-value:`true`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`Whether output should be done in hdf5, else VTK.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``hdf5_compression_level``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`6`
   * - **Allowed Range:**
     - :allowed-range:`1:9`
   * - **Description:**
     - :param-description:`Compression level of hdf5 output, note that you need a compression library installed for this to do anything.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``hdf5_chunk_size``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`1000`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`Size of chunks, can significantly impact performance, only change if you're actively profiling IO on a new system.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``volume_output_cell_variables``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform volume output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``plane_surface_output_cell_variables``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform plane surface output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``sphere_surface_output_cell_variables``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform sphere surface output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_minmax``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform min/max reduction output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_norm2``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform L2 norm reduction output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``scalar_output_integral``
   * - **Type:**
     - :data-type:`list of keywords`
   * - **Default Value:**
     - :default-value:`[]`
   * - **Allowed Range:**
     - :allowed-range:`List containing any number of evolved or auxiliary cell-centered variables.`
   * - **Description:**
     - :param-description:`Which variables to perform integral reduction output for.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_extra_quantities``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`Perform output of extra quantities (rank_id, ref_level, coords)?`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``n_output_planes``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`3`
   * - **Allowed Range:**
     - :allowed-range:`0:`
   * - **Description:**
     - :param-description:`How many planes to output, first 3 are reserved (xy,xz,yz), max is 10.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_x_origin_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`x coordinate origin of nth plane (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_y_origin_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`y coordinate origin of nth plane (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_z_origin_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`z coordinate origin of nth plane (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_x_normal_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`x coordinate normal of nth plane (n from 0 to 9), for plane #2 default is 1..`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_y_normal_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`y coordinate normal of nth plane (n from 0 to 9) for plane #1 default is 1..`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_z_normal_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`z coordinate normal of nth plane (n from 0 to 9) for plane #0 default is 1..`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_plane_name_[0-9]``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`true, false`
   * - **Allowed Range:**
     - :allowed-range:`Anything`
   * - **Description:**
     - :param-description:`For first three planes default is xy, xz, yz.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``n_output_spheres``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`3`
   * - **Allowed Range:**
     - :allowed-range:`0:`
   * - **Description:**
     - :param-description:`How many spheres to output, first 3 are reserved (xy,xz,yz), max is 10.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_x_center_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`x coordinate center of nth sphere (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_y_center_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`y coordinate center of nth sphere (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_z_center_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`z coordinate center of nth sphere (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_radius_[0-9]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Radius of nth sphere (n from 0 to 9).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_name_[0-9]``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`true, false`
   * - **Allowed Range:**
     - :allowed-range:`Anything`
   * - **Description:**
     - :param-description:`true, false`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_sphere_tracking_[0-9]``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`none`
   * - **Allowed Range:**
     - :allowed-range:`none, (max_density, min_alp)`
   * - **Description:**
     - :param-description:`Tracking criterion for nth sphere (work in progress).`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the IO module to their default value.

.. code-block:: yaml

   IO: {
     volume_output : true, 
     surface_output: true,
     
     volume_output_every : 10, 
     plane_surface_output_every: 10,
     sphere_surface_output_every: 10,
     scalar_output_every: 10,
     info_output_every: 10,
   
     info_output_max_reductions: [],
     info_output_min_reductions: [],
     info_output_norm2_reductions: [],
   
     log_output_base_filename: "grace_log",
     log_output_base_directory: "./logs", 
     one_log_per_rank: false,
   
     volume_output_base_filename: "volume_out",
     volume_output_base_directory: "output_volume",
   
     surface_output_base_filename: "surface_out",
     surface_output_base_directory: "output_surface",
   
     scalar_output_base_filename: "",
     scalar_output_base_directory: "output_scalar",
   
     output_use_hdf5: true,
     hdf5_compression_level: 6,
     hdf5_chunk_size: 1000,
   
     volume_output_cell_variables : [],
     plane_surface_output_cell_variables: [],
     sphere_surface_output_cell_variables: [],
     scalar_output_minmax: [],
     scalar_output_norm2: [],
     scalar_output_integral: [],
   
     output_extra_quantities: false,
   
     n_output_planes: 3,
     output_plane_x_origin_0: 0,
     output_plane_x_origin_1: 0,
     output_plane_x_origin_2: 0,
     output_plane_x_origin_3: 0,
     output_plane_x_origin_4: 0,
     output_plane_x_origin_5: 0,
     output_plane_x_origin_6: 0,
     output_plane_x_origin_7: 0,
     output_plane_x_origin_8: 0,
     output_plane_x_origin_9: 0,
   
     output_plane_y_origin_0: 0,
     output_plane_y_origin_1: 0,
     output_plane_y_origin_2: 0,
     output_plane_y_origin_3: 0,
     output_plane_y_origin_4: 0,
     output_plane_y_origin_5: 0,
     output_plane_y_origin_6: 0,
     output_plane_y_origin_7: 0,
     output_plane_y_origin_8: 0,
     output_plane_y_origin_9: 0,
   
     output_plane_z_origin_0: 0,
     output_plane_z_origin_1: 0,
     output_plane_z_origin_2: 0,
     output_plane_z_origin_3: 0,
     output_plane_z_origin_4: 0,
     output_plane_z_origin_5: 0,
     output_plane_z_origin_6: 0,
     output_plane_z_origin_7: 0,
     output_plane_z_origin_8: 0,
     output_plane_z_origin_9: 0,
   
   
     output_plane_x_normal_0: 0,
     output_plane_x_normal_1: 0,
     output_plane_x_normal_2: 1,
     output_plane_x_normal_3: 0,
     output_plane_x_normal_4: 0,
     output_plane_x_normal_5: 0,
     output_plane_x_normal_6: 0,
     output_plane_x_normal_7: 0,
     output_plane_x_normal_8: 0,
     output_plane_x_normal_9: 0,
   
     output_plane_y_normal_0: 0,
     output_plane_y_normal_1: 1,
     output_plane_y_normal_2: 0,
     output_plane_y_normal_3: 0,
     output_plane_y_normal_4: 0,
     output_plane_y_normal_5: 0,
     output_plane_y_normal_6: 0,
     output_plane_y_normal_7: 0,
     output_plane_y_normal_8: 0,
     output_plane_y_normal_9: 0,
   
     output_plane_z_normal_0: 1,
     output_plane_z_normal_1: 0,
     output_plane_z_normal_2: 0,
     output_plane_z_normal_3: 0,
     output_plane_z_normal_4: 0,
     output_plane_z_normal_5: 0,
     output_plane_z_normal_6: 0,
     output_plane_z_normal_7: 0,
     output_plane_z_normal_8: 0,
     output_plane_z_normal_9: 0,
   
   
     output_plane_name_0: "xy",
     output_plane_name_1: "xz",
     output_plane_name_2: "yz",
     output_plane_name_3: "3",
     output_plane_name_4: "4",
     output_plane_name_5: "5",
     output_plane_name_6: "6",
     output_plane_name_7: "7",
     output_plane_name_8: "8",
     output_plane_name_9: "9",
   
     n_output_spheres: 0,
     output_sphere_x_center_0: 0,
     output_sphere_x_center_1: 0,
     output_sphere_x_center_2: 0,
     output_sphere_x_center_3: 0,
     output_sphere_x_center_4: 0,
     output_sphere_x_center_5: 0,
     output_sphere_x_center_6: 0,
     output_sphere_x_center_7: 0,
     output_sphere_x_center_8: 0,
     output_sphere_x_center_9: 0,
   
     output_sphere_y_center_0: 0,
     output_sphere_y_center_1: 0,
     output_sphere_y_center_2: 0,
     output_sphere_y_center_3: 0,
     output_sphere_y_center_4: 0,
     output_sphere_y_center_5: 0,
     output_sphere_y_center_6: 0,
     output_sphere_y_center_7: 0,
     output_sphere_y_center_8: 0,
     output_sphere_y_center_9: 0,
   
     output_sphere_z_center_0: 0,
     output_sphere_z_center_1: 0,
     output_sphere_z_center_2: 0,
     output_sphere_z_center_3: 0,
     output_sphere_z_center_4: 0,
     output_sphere_z_center_5: 0,
     output_sphere_z_center_6: 0,
     output_sphere_z_center_7: 0,
     output_sphere_z_center_8: 0,
     output_sphere_z_center_9: 0,
   
     output_sphere_radius_0: 1,
     output_sphere_radius_1: 1,
     output_sphere_radius_2: 1,
     output_sphere_radius_3: 1,
     output_sphere_radius_4: 1,
     output_sphere_radius_5: 1,
     output_sphere_radius_6: 1,
     output_sphere_radius_7: 1,
     output_sphere_radius_8: 1,
     output_sphere_radius_9: 1,
   
     output_sphere_name_0: "0",
     output_sphere_name_1: "1",
     output_sphere_name_2: "2",
     output_sphere_name_3: "3",
     output_sphere_name_4: "4",
     output_sphere_name_5: "5",
     output_sphere_name_6: "6",
     output_sphere_name_7: "7",
     output_sphere_name_8: "8",
     output_sphere_name_9: "9",
   
     output_sphere_tracking_0: "none",
     output_sphere_tracking_1: "none",
     output_sphere_tracking_2: "none",
     output_sphere_tracking_3: "none",
     output_sphere_tracking_4: "none",
     output_sphere_tracking_5: "none",
     output_sphere_tracking_6: "none",
     output_sphere_tracking_7: "none",
     output_sphere_tracking_8: "none",
     output_sphere_tracking_9: "none"
   
   
   }


