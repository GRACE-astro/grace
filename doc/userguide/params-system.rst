.. GRACE documentation System Parameters file

.. _system-parameters:

System Parameters
*****************

This page contains a full reference of all parameters available for the system module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``master_rank``
   * - **Type:**
     - :data-type:`int`
   * - **Default Value:**
     - :default-value:`0`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`The master rank performs output.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``console_log_level``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`info`
   * - **Allowed Range:**
     - :allowed-range:`trace, debug, info, warn, error, critical`
   * - **Description:**
     - :param-description:`The minimum severity level that is printed to console.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``file_log_level``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`debug`
   * - **Allowed Range:**
     - :allowed-range:`trace, debug, info, warn, error, critical`
   * - **Description:**
     - :param-description:`The minimum severity level that is printed to logfiles.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``flush_logs_based_on_severity``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`If true, message above a certain severity will flush the log buffers and cause all output to be printed.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``flush_severity_level``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`error`
   * - **Allowed Range:**
     - :allowed-range:`trace, debug, info, warn, error, critical`
   * - **Description:**
     - :param-description:`If ` ``flush_logs_based_on_severity`` :param-description:`, this is the severity level that triggers a flush.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``flush_logs_based_on_time``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`If true, log buffers will be flushed every n seconds printing all messages in the queue.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``flush_time``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`10.0`
   * - **Allowed Range:**
     - :allowed-range:`0:*`
   * - **Description:**
     - :param-description:`If ` ``flush_logs_based_on_time`` :param-description:`, this is the number of seconds between flushes.`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the system module to their default value.

.. code-block:: yaml

   system: {
     master_rank: 0,
     console_log_level: "info",
     file_log_level: "debug",
     flush_logs_based_on_severity: false,
     flush_severity_level: "error",
     flush_logs_based_on_time: false,
     flush_time: 10
   }


