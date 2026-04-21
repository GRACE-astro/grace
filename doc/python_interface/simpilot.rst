.. _gracepy-simpilot:

Simpilot
=========

``simpilot`` is a lean tool designed to aid the management of simulations with GRACE on HPC clusters. It maintains a database of known machines together with their environment files and submission script templates, and provides commands to create, submit, and monitor simulations.


Setup
*******

On first use, ``simpilot`` will interactively configure itself. The configuration is stored in ``~/.simpilot/`` and includes:

- ``known_machines.yaml`` — registry of available HPC platforms
- ``user_settings.yaml`` — user preferences (e.g. default email for notifications)
- ``machines/`` — per-machine YAML configuration files
- ``submitscripts/`` — submission script templates
- ``env_files/`` — environment files (module loads, library paths)
- ``active_simulations/`` — records of managed simulations

The tool reads two environment variables:

- ``SIMPLOT_BASEDIR`` — base directory where simulations are created
- ``GRACE_HOME`` — path to the GRACE installation


Machine Configuration
***********************

Each machine is described by a YAML file specifying its scheduler, hardware, submission template, and environment file:

.. code-block:: yaml

    name: frontier
    scheduler: slurm
    max_walltime: "24:00:00"
    backend: hip
    mem_per_node: 512       # GB
    cpu_per_node: 64
    gpu_per_node: 8
    default_queue: batch
    queues:
      - batch
      - debug
    submission_script: "frontier-gpu.sub"
    environment_file: "frontier-gpu.sh"

The ``environment_file`` is a shell script that loads modules and sets library paths. It is sourced in the submission script via the ``@ENV_FILE@`` placeholder. Both the submission template and environment file are copied into each simulation's ``config/`` directory at creation time, and into each ``restart_NNNN/`` directory at submission time, making every run self-contained.

Currently supported schedulers:

- **SLURM** — full support (submit, chain, cancel, status query)
- **PBS** — planned but not yet implemented

Creating a Simulation
***********************

.. code-block:: bash

    simpilot create --simname <simname> --machine <machine> \
                    --parameter_file <parfile> --executable <executable>

This sets up a directory structure under the simulation base directory:

.. code-block:: text

    <simname>/
    ├── config/
    │   ├── grace             # copy of the executable
    │   ├── machine.yaml      # machine configuration snapshot
    │   ├── parfile/          # copy of the parameter file
    │   ├── submission/       # submission script template
    │   ├── env/              # environment file
    │   └── status.yaml       # job history (restart_id, job_id)
    └── restart_0000/         # first run segment

The parameter file and executable path are recorded so that subsequent submissions are self-contained.


Submitting a Simulation
*************************

.. code-block:: bash

    simpilot submit --simname <simname> --walltime <hh:mm:ss> --nodes <n_nodes>

If the machine configuration includes ``mem_per_node``, ``cpu_per_node``, and ``gpu_per_node``, this is sufficient. Otherwise these values must be specified manually.

**Job chaining**: if a previous job for this simulation is still queued or running, ``simpilot`` automatically submits the new job with a dependency constraint so that it starts only after the previous one finishes. This enables seamless multi-segment runs without manual intervention.

Additional options are available via ``simpilot submit --help``, including queue selection and email notifications.


Python API
************

The same functionality is available programmatically through the ``grace_pilot`` package:

.. code-block:: python

    from grace_pilot.machine import machine
    from grace_pilot.simulation import simulation
    from grace_pilot.simpilot import simpilot

    # Load machine configuration
    m = machine("path/to/machine_config.yaml")

    # Create and manage a simulation
    sp = simpilot()
    sp.create_new_simulation(
        simname="bns_run",
        simpath="/scratch/bns_run",
        _machine=m,
        executable="/path/to/grace",
        parameter_file="parfile.yaml",
    )
