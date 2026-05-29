.. _grace-singleton-holder:

The singleton holder
====================

Several pieces of state in GRACE are unavoidably global: the MPI
communicator, the Kokkos runtime, the parameter store, the p4est
forest, the EOS instance, the variable list, the coordinate system.
GRACE manages all of them through a single, policy-based singleton
template.  This chapter describes that template, how it is used, and
the few sharp edges that have come up in practice.


The pattern
-----------

The implementation in ``include/grace/utils/singleton_holder.hh`` is
a direct descendant of the ``SingletonHolder`` template from
Alexandrescu's *Modern C++ Design* and its accompanying Loki library
[Alexandrescu2001]_.  We kept its three central ideas:

1. The singleton is wrapped in a *holder* template, parameterised on
   the held type and on a *creation policy*.  Allocation strategy
   (``new``/``delete``, placement-new with a custom allocator,
   pool-backed, etc.) is selected at the holder declaration site
   without touching the held type.
2. Each held type declares a *longevity*, an integer priority that
   controls the order in which singletons are torn down at process
   exit.  Lower-priority objects outlive higher-priority ones, so
   downstream consumers go first and upstream runtimes go last.
3. Construction is lazy: the first call to ``get()`` constructs the
   instance via the creation policy and registers it for destruction
   at the appropriate point in the longevity sequence.


What we changed from Loki
-------------------------

- C++20 throughout.  Default-constructibility is detected with
  ``if constexpr`` and ``std::is_default_constructible``; perfect
  forwarding via parameter packs replaces Loki's variadic emulation.
- The "touch after destroy" path that Loki handles with its
  ``Phoenix`` policy is replaced by an explicit ``deleted_`` flag
  and an ``ASSERT``.  If a singleton is accessed after its lifetime
  has ended we want to see it, not silently resurrect the object.
- The creation policy is taken as a template-template parameter so
  the held type only has to be spelled once.


Usage
-----

A type ``T`` is made a singleton by giving it a private (or
protected) constructor and destructor, friending the holder, and
exposing a public static ``longevity`` constant of type
``unique_objects_lifetimes``:

.. code-block:: cpp

   class my_resource {
       friend class utils::singleton_holder<my_resource> ;

       my_resource() = default ;
       ~my_resource() = default ;

     public:
       static constexpr auto longevity = SYSTEM_UTILITIES ;
       void do_something() ;
   } ;

Access is by reference only — because the destructor is
inaccessible, a copy is a compile error:

.. code-block:: cpp

   auto& res = utils::singleton_holder<my_resource>::get() ;  // OK
   auto  res = utils::singleton_holder<my_resource>::get() ;  // error

Types that are not default-constructible must be initialised
explicitly before the first ``get()``:

.. code-block:: cpp

   utils::singleton_holder<my_resource>::initialize(arg1, arg2) ;
   auto& res = utils::singleton_holder<my_resource>::get() ;

Most GRACE subsystems re-export their access through a thin static
``get()`` member of the held type itself, so call sites read

.. code-block:: cpp

   auto& vars = grace::variable_list::get() ;

rather than spelling the holder template explicitly.  This is
cosmetic; the underlying mechanism is the same.


Longevity and finalisation
--------------------------

Longevities are declared in
``include/grace/utils/lifetime_tracker.hh`` as the enumerator
``unique_objects_lifetimes``.  At exit, singletons are torn down in
order of decreasing longevity: the most "downstream" objects first,
the most "upstream" runtimes last, so that no singleton ever finds
its dependencies already destroyed.

The list is hand-maintained.  When you add a new singleton, choose
its longevity by asking *which existing singletons must outlive me?*
— the new entry goes immediately below the latest such dependency.
In broad strokes:

- pure utilities that nothing else depends on:
  ``SYSTEM_UTILITIES``
- foreign runtimes (MPI, profiling, Kokkos, p4est):
  ``MPI_RUNTIME``, ``GRACE_PROFILING_RUNTIME``,
  ``KOKKOS_RUNTIME``, ``P4EST_RUNTIME``
- compute resources allocated against those runtimes:
  ``DEVICE_RESOURCES``
- physics-aware singletons (parameter store, EOS, variable list,
  coordinate system, forest): later in the list.


Pitfalls
--------

- **Holding the result of** ``get()`` **by value** compiles only if
  the destructor is public, which would defeat the purpose of the
  pattern.  Always assign to ``auto&``.
- **Calling** ``get()`` **after** ``destroy()`` triggers an
  assertion rather than re-creating the object.  This is intentional
  but is the most common shutdown bug for new code paths that drive
  teardown from non-standard places (signal handlers, error paths,
  Catch2 fixtures that re-enter ``initialize``).
- **Forgetting** ``T::longevity`` compiles only if you happen to
  have another symbol of that name in scope; the resulting
  destruction order is then undefined.  Always pick an entry from
  ``unique_objects_lifetimes``.
- **Singletons are not thread-safe**.  GRACE assumes a single host
  thread drives MPI ranks and Kokkos device kernels; singleton
  initialisation happens before any worker threads start.  If you
  introduce host-side multithreading, ``get()`` and ``initialize()``
  will need a guard.


.. [Alexandrescu2001] A. Alexandrescu, *Modern C++ Design: Generic
   Programming and Design Patterns Applied*, Addison-Wesley (2001).
   Chapter 6 ("Implementing Singletons") covers the policy-based
   ``SingletonHolder`` and longevity machinery that inspired this
   implementation.  The Loki library implements these ideas; see
   `<http://loki-lib.sourceforge.net/>`_.
