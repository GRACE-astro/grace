.. GRACE documentation EOS Parameters file

.. _eos-parameters:

EOS Parameters
**************

This page contains a full reference of all parameters available for the eos module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``eos_type``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`hybrid`
   * - **Allowed Range:**
     - :allowed-range:`hybrid, tabulated`
   * - **Description:**
     - :param-description:`EOS type.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``cold_eos_type``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`piecewise_polytrope`
   * - **Allowed Range:**
     - :allowed-range:`piecewise_polytrope, tabulated`
   * - **Description:**
     - :param-description:`For hybrid EOSs: cold EOS type.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``ye_atmosphere``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`The electron fraction value in the atmosphere.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``temp_atmosphere``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`The temperature value in the atmosphere.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``rho_atmosphere``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.e-14`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`The rest-mass density value in the atmosphere.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``atm_is_beta_eq``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`true`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`If true, the atmosphere is set to be in beta-equilibrium (overwrites ye_atmosphere).`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``extend_table_high``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`If true, out of bounds densities along maximum table value are extrapolated.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``eps_maximum``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.e+05`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`Maximum allowed specific internal energy.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``gamma_th``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.8`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For hybrid EOSs: the value of Gamma thermal.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``piecewise_polytrope``
   * - **Type:**
     - :data-type:`dictionary`
   * - **Default Value:**
     - :default-value:`--`
   * - **Allowed Range:**
     - :allowed-range:`--`
   * - **Description:**
     - :param-description:`For piecewise polytropic hybrid EOSs: piecewise polytrope settings.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``piecewise_polytrope["n_pieces"]``
   * - **Type:**
     - :data-type:`unsigned int`
   * - **Default Value:**
     - :default-value:`7`
   * - **Allowed Range:**
     - :allowed-range:`1:*`
   * - **Description:**
     - :param-description:`For piecewise polytropic cold EOSs: number of segments.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``piecewise_polytrope["gammas"]``
   * - **Type:**
     - :data-type:`list of double`
   * - **Default Value:**
     - :default-value:`[ 1.58425 , 1.28733     , 0.62223     , 1.35692     , 3.005       , 2.988       , 2.851]`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For piecewise polytropic cold EOSs: Gamma of each segment. Length of this list must be n_pieces, default is SLy4 parametrization by Read et al.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``piecewise_polytrope["rhos"]``
   * - **Type:**
     - :data-type:`list of double`
   * - **Default Value:**
     - :default-value:`[ 3.951156e-11, 6.125960e-07, 4.254672e-06, 2.367449e-04, 8.114721e-04, 1.619100e-03]`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For piecewise polytropic cold EOSs: rho of each segment in code units. Length of this list must be n_pieces-1 since the first is always 0, default is SLy4 parametrization by Read et al.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``piecewise_polytrope["kappa_0"]``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.685819e+02`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`For piecewise polytropic cold EOSs: first polytropic constant, the rest are calculated based on junction conditions.`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the eos module to their default value.

.. code-block:: yaml

   eos: {
     eos_type: "hybrid",
     cold_eos_type: "piecewise_polytrope",
   
     ye_atmosphere: 0,
     temp_atmosphere: 0,
     rho_atmosphere: 1.e-14,
     eps_maximum: 1.e+05,
     gamma_th: 1.8,
   
     atm_is_beta_eq: true,
     extend_table_high: false,
   
     piecewise_polytrope: {
       n_pieces: 7,
       gammas: [ 1.58425 , 1.28733     , 0.62223     , 1.35692     , 3.005       , 2.988       , 2.851],
       rhos:   [ 3.951156e-11, 6.125960e-07, 4.254672e-06, 2.367449e-04, 8.114721e-04, 1.619100e-03],
       kappa_0: 1.685819e+02
     }
   }


