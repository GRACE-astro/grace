.. _grace-config-parser:

Configuration and parameter schemas
===================================

GRACE has a single, unified way to describe and access runtime
parameters: a set of YAML schema files, eagerly applied at startup,
fronted by a singleton parameter store.  Every parameter in the code
— from atmosphere floors to AMR refinement criteria to logging behaviour — 
passes through the same machinery, validated once and then accessed by simple 
key lookups.

This chapter describes the system from the developer's perspective:
how parameters are declared, how validation works, how to add a new
module, and the few pitfalls that have come up in practice.


The two-tier view
-----------------

Every GRACE module declares its parameters in a *schema file* under
``parameters/<module>.yaml``.  The schema describes each parameter's
type, default value, range, and a human-readable description.  At
runtime, the user supplies a *user config* — a regular YAML file
that overrides whichever schema defaults the run actually needs to
change.

Schema and user config follow the same syntactic structure, so a
user file looks like a sparse subset of the corresponding schemas
with the metadata fields stripped:

.. code-block:: yaml

   # parameters/grmhd.yaml (schema, abridged)
   grmhd:
     type: schema
     atmosphere:
       type: schema
       rho_fl:
         type: double
         range: "[0,*)"
         default: 1.0e-14
         description: >
           Rest-mass density floor.

.. code-block:: yaml

   # user_config.yaml
   grmhd:
     atmosphere:
       rho_fl: 1.0e-12   # override

After startup, both views are merged: every parameter declared in any
schema is guaranteed present in the live parameter tree, with either
the user's override or the schema's default.


Supported parameter types
-------------------------

The ``type:`` field controls how the value is parsed and validated:

==================  =================================================
``type:``           Notes
==================  =================================================
``double``          ``range`` is required; ``[..]`` inclusive, ``(..)`` exclusive, ``*`` for ±∞.
``int``             Same range semantics as ``double``.
``unsigned int``    Same; range lower bound implicit at 0.
``bool``            ``true`` / ``false``.
``keyword``         Requires ``allowed: [...]``.  Value is checked against the list.
``string``          Free-form; no validation.
``schema``          A nested sub-tree.  Parameters inside are addressed by chained keys.
``list``            Requires ``item_type`` (one of the above) and optionally ``item_schema`` for ``node`` items, ``min_items`` for length checks.
==================  =================================================

The validation logic lives in ``include/grace/config/yaml_helpers.hh``
and ``src/config/yaml_helpers.cpp``; the ``numeric_range<T>`` template
parses the bracket notation once and caches the bounds.


Eager defaults
--------------

The single most important invariant of the parameter system is that
**every parameter declared in a schema is present in the live tree
after startup**.  Call sites never need a default-value fallback,
because the schema has already done that job.

This is achieved by
``config_parser_impl_t::set_parameters_to_default_value()``, which
runs once during ``grace::config_parser::initialize(parfile)``:

1. Iterate the ``code_modules`` array (declared in
   ``include/grace/config/code_modules.h.in``; populated at CMake
   configure time from the modules enabled in the build).
2. For each module, load its schema from
   ``parameters/<module>.yaml``.
3. Call ``traverse_section()`` to walk the schema, fill missing user
   values with schema defaults, and validate types, ranges, and
   keywords against the schema.
4. Emit a warning for every key in the user config that is not
   recognised by any schema (typo guard).

The result is a tree in which every documented parameter has a
sensible value and every undocumented user key has been flagged.


Accessing parameters
--------------------

Parameters are read with the variadic template ``get_param<T>``:

.. code-block:: cpp

   auto const rho_fl  = grace::get_param<double>("grmhd","atmosphere","rho_fl") ;
   auto const cfl     = grace::get_param<double>("evolution","cfl_factor") ;
   auto const id_type = grace::get_param<std::string>("grmhd","id_type") ;

The key sequence walks the YAML tree; the final node is converted to
the requested type via ``yaml-cpp``'s ``as<T>()``.  Because defaults
are eager there is no overload for fallback values — if the call
succeeds, the value comes from either the user file or the schema;
if it fails, the parameter wasn't declared in any schema and the
build is wrong.

For convenience, modules whose parameters are dense (atmosphere,
excision, C2P, FOFC, Riemann solver, …) wrap their reads into a
plain struct populated by a single helper function:

.. code-block:: cpp

   atmo_params_t    get_atmo_params() ;
   c2p_params_t     get_c2p_params() ;
   fofc_params_t    get_fofc_params() ;
   riemann_params_t get_riemann_params() ;

Subsystems then read the struct once at construction time and pass
it by value into Kokkos kernels via the enclosing class.  This keeps
the hot path free of YAML lookups.


Adding a new module
-------------------

1. Create ``parameters/new_module.yaml`` with a top-level entry of
   ``type: schema`` and the desired parameters underneath.
2. Add the module to ``include/grace/config/code_modules.h.in``,
   guarded by the relevant ``#ifdef`` if the module is conditional
   on a CMake option.
3. Access parameters via ``get_param<T>("new_module", "param_name")``.
   Defaults and validation are automatic; nothing else needs to be
   wired.


Adding a new parameter to an existing module
--------------------------------------------

1. Add the parameter to its module's schema with ``type``,
   ``default``, and ``range`` (and ``description``, even if it's
   short).
2. Read it where needed via ``get_param``.
3. Done.  The default makes the parameter optional for users; the
   schema is the single source of truth and is harvested into the
   user-guide table at documentation build time
   (``generate_configure_options_table.py.in``).


Pitfalls
--------

- **Forgetting to add a new module to** ``code_modules.h.in``
  silently disables both the defaults and the validation for that
  module: ``get_param`` then sees an empty subtree and ``as<T>()``
  throws an obscure ``yaml-cpp`` error.  If you see "bad conversion"
  early in startup, this is the first thing to check.
- **Calling** ``get_param`` **before** ``config_parser::initialize``
  fires the singleton lifetime assertion described in
  :ref:`grace-singleton-holder` — the parser hasn't been built yet.
  Module constructors that need parameters should always be invoked
  after the system initialisation sequence has reached the parser.
- **Mismatched** ``type:`` **and** ``default:``.  YAML doesn't quote
  numerics, so a schema entry that declares ``type: double`` with
  ``default: 1`` works, but ``type: int`` with ``default: 1.0`` will
  trip validation.  Use the right literal form.
- **Stale defaults after a parameter rename**.  Renaming a parameter
  in the schema *and* its access site is half the work; user configs
  that referred to the old name are now silently unused and trigger
  a warning rather than a hard error.  Watch the warning log on the
  first run after a rename.
- **Hot-path lookups**.  ``get_param`` is fine at startup but is not
  a constant expression; do not call it from inside a kernel or a
  per-cell loop.  Read once into a local or a struct and capture by
  value.
- **List parameters with mixed item types** are not supported.  If
  you find yourself wanting one, the right move is a list of
  ``item_type: node`` whose schema is a small struct, not a list of
  ``double | string``.
