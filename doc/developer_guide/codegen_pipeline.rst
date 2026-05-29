.. _grace-codegen-pipeline:

The codegen pipeline
====================

Several files in ``include/grace/physics/`` end in
``_subexpressions.hh`` and contain bodies that look like a SymPy
common-subexpression-elimination dump:

.. code-block:: cpp

   static void KOKKOS_INLINE_FUNCTION
   z4c_get_det_conf_metric(
       const double gtdd[6],
       double * __restrict__ detg
   )
   {
       *detg = gtdd[0]*gtdd[3]*gtdd[5] - gtdd[0]*((gtdd[4])*(gtdd[4]))
             - ((gtdd[1])*(gtdd[1]))*gtdd[5] + 2*gtdd[1]*gtdd[2]*gtdd[4]
             - ((gtdd[2])*(gtdd[2]))*gtdd[3];
   }

That is because they are SymPy CSE output.  The bodies are not
maintained by hand: they are emitted by Jupyter notebooks living
in the ``GRACEpy`` repository, using a small set of helpers in
``GRACEpy/src/codegen/codegen_utils.py``, and written directly
into the GRACE include tree.  This chapter describes the pipeline
end to end so a contributor adding new physics knows where to make
the change and what to expect from the printer.


What is generated
-----------------

The current set of generated headers in
``include/grace/physics/`` is:

- ``fd_subexpressions.hh`` — finite-difference stencils and
  related operators.
- ``grmhd_subexpressions.hh`` — GRMHD primitive / conservative
  helpers, flux pieces, magnetic-field helpers.
- ``m1_subexpressions.hh`` — M1 closure expressions.
- ``z4c_subexpressions.hh`` — the Z4c RHS and related tensorial
  operations (Ricci, contracted Christoffels, ...).
- ``z4c_subexpressions_regrouped.hh`` — a parity-equivariant
  regrouped variant of the above; see "Symmetry equivariance"
  below.

Other long-form expressions in the physics directory
(Riemann-solver internals, initial-data analytic helpers) follow
the same pipeline.


Source of truth: the notebooks
------------------------------

The authoritative source for each generated header is the
corresponding notebook, organised
by physics module:

- ``metric_evol/`` — Z4c.  Z4c specifically:
  ``metric_evol/Z4c_W.py`` (and ``.ipynb``).
- ``hydro_evol/`` — GRMHD flux/RHS helpers.
- ``M1/`` — M1 radiation closure.
- ``InitialData/`` — analytic initial-data helpers (Bondi flow,
  Kerr-Schild metric, ...).

Each notebook follows the same pattern: build the symbolic
expressions for the physics quantities of interest using SymPy,
collect them into a dictionary of named outputs, hand the
dictionary to a helper that emits a C function declaration and
body, and write the result into the GRACE source tree.

The header file is therefore a **derived artefact**.  It is
checked into version control so that GRACE builds without
requiring the notebook to be re-run, but the canonical source is
the notebook.  Hand-edits to a ``_subexpressions.hh`` are
**not durable**: the next regeneration of the notebook will
overwrite them.


Codegen helpers
---------------

The thin layer between SymPy and the GRACE include tree lives in
``GRACEpy/src/codegen/codegen_utils.py``.  The pieces a
notebook author touches are:

**``MyPrinter``** — a ``sympy.printing.c.C99CodePrinter`` subclass.
Customisations:

- ``_print_Pow`` rewrites integer powers as repeated multiplies
  (``x**3`` → ``(x)*(x)*(x)``) and ``x ** (-1/2)`` as
  ``1.0 / sqrt(x)``.  This matters for both performance (avoids
  ``pow`` for small integer powers) and for IEEE round-off
  predictability.
- ``_print_Piecewise`` emits piecewise expressions as nested C
  ternaries so they remain a single expression in the C source
  (compiler can hoist common terms).
- ``_print_Add`` is **not overridden**.  Addition order
  falls through to ``C99CodePrinter``'s default canonical
  SymPy ordering.  Consequences for symmetry are discussed
  below.

**``MyPyPrinter``** — the NumPy variant of ``MyPrinter``, used
when a notebook needs a Python reference implementation of the
same expressions (for cross-checking, or for use in Python tests).

**``make_body(exprs, printer, outputs, layout, cse_order,
cse_optims, ...)``** — runs

.. code-block:: python

   subexprs, reduced = sp.cse(exprs,
                              optimizations=cse_optims,
                              order=cse_order,
                              ignore=cse_ignore)

then iterates the resulting list and emits each subexpression as
``double xN = ...;`` followed by the reduced output writes.  The
default ``cse_order='canonical'`` and ``cse_optims='basic'`` are
what almost all GRACE notebooks use.

**``make_function(exprs, printer, name, ABI, outputs, ...,
add_to_output=False)``** — wraps ``make_body`` with a function
declaration: argument list derived from ``ABI``, output
destinations from ``outputs_ABI``, optional template-argument
prefix, optional read-modify-write semantics via
``add_to_output``.  This is the entry point a notebook actually
calls.

**``make_function_py``** — the matching Python emitter using
``MyPyPrinter``.


How a new function lands in a header
------------------------------------

The mechanics, in the simplest case:

.. code-block:: python

   from codegen_utils import MyPrinter, make_function
   import sympy as sp

   # 1. Define symbols and build expressions
   gtdd = sp.IndexedBase('gtdd', shape=(6,))
   det  = (gtdd[0]*gtdd[3]*gtdd[5]
           - gtdd[0]*gtdd[4]**2
           - gtdd[1]**2*gtdd[5]
           + 2*gtdd[1]*gtdd[2]*gtdd[4]
           - gtdd[2]**2*gtdd[3])

   # 2. Describe the function ABI (argument names + types)
   ABI = {"gtdd": ("double", [6])}
   outputs = {"detg": det}
   outputs_ABI = {"gtdd": ("double", None)} # This indicates a scalar double

   # 3. Emit
   printer = MyPrinter()
   src = make_function(
       outputs.values(),
       printer,
       name="z4c_get_det_conf_metric",
       ABI=ABI,
       outputs=list(outputs),
       outputs_ABI=outputs_ABI,
   )

   # 4. Write into the GRACE include tree
   with open("…/include/grace/physics/z4c_subexpressions.hh", "a") as f:
       f.write(src)

In practice the notebooks build many expressions at once, group
them by output array, and call ``make_function`` once per logical
operation (one per Z4c RHS variable, etc.).  The notebook then
opens the destination header in write mode and dumps a banner +
all the functions in one pass, so the file's contents are a
single self-consistent snapshot of the notebook state.


Symmetry equivariance and the ``regrouped`` variant
---------------------------------------------------

The CSE pass and the printer have an important interaction with
GRACE's symmetry-equivariance goals (see the design principles
chapter).  Two facts are load-bearing:

- ``sp.cse(...)`` produces subexpressions in a canonical order
  that is *unaware* of which symbols are partners under a
  discrete symmetry such as :math:`z \to -z`.  Two CSE temporaries
  that ought to be partners under reflection may end up named
  ``x12`` and ``x47`` and computed in different positions in the
  emitted code, so a sum of "left half" and "right half"
  contributions can land in an order whose IEEE round-off is not
  bit-mirror under the discrete symmetry.
- ``MyPrinter._print_Add`` falls back to ``C99CodePrinter``'s
  canonical lex order.  With ``-fno-associative-math`` the
  compiler respects this order, which means *whichever* ordering
  SymPy picks is what the C code uses.  A parity-equivariant
  emission has to be arranged at the SymPy level, not at the
  printer level.

The ``z4c_subexpressions_regrouped.hh`` variant is the result of
the latter.  Its source notebook decomposes the Z4c RHS into
named tensor-level chunks (typically one chunk per parity class
under the discrete symmetries that BNS evolutions care about),
runs CSE on each chunk independently, and accumulates them at the
C level with explicit pair-symmetric ordering.  The regrouped
header is mathematically identical to ``z4c_subexpressions.hh``
but produces bit-mirror output under reflection where the plain
header does not.

The ``regrouped`` form is used in production for BNS runs where
mirror symmetry needs to be preserved to floating-point floor.
The plain ``z4c_subexpressions.hh`` is kept available because it
is a much shorter file, useful for quick experimentation and for
single-star tests where symmetry is not the binding constraint.


When to regenerate
------------------

You must regenerate the relevant ``_subexpressions.hh`` whenever:

- the underlying physics expressions change (a new RHS term, a
  changed sign convention, a different gauge);
- the ABI of a generated function changes (added argument, new
  template parameter);
- the SymPy notebook itself is edited for any reason — even a
  cosmetic refactor — because CSE temporaries are renumbered in
  ways that depend on the input expression order;
- the printer is updated in ``codegen_utils.py`` and you want the
  effect of the new printer to land in the headers.

You should **not** hand-edit a ``_subexpressions.hh`` file as the
primary fix path.  An emergency one-off patch for a bug found
during a deadline is fine — but the change must also be
committed back into the source notebook within the same PR, or
the next regeneration silently undoes it.

A practical workflow for a notebook change:

1. Edit the notebook.
2. Re-run the notebook from top to bottom.  The final cell
   should write the header into
   ``…/grace-private-src/include/grace/physics/`` (paths are
   notebook-local; check the writer cell).
3. ``git diff`` the affected header to confirm the change is what
   you expected.  Pay attention to CSE temporary renumbering — a
   real one-line physics change can produce a 50-line diff in
   the generated header.
4. Rebuild GRACE and run the relevant unit tests.  CSE-renumber
   diffs that don't change semantics will keep tests green; if a
   test goes red, it almost always means the underlying SymPy
   expression has a real difference, not a printer artefact.


Pitfalls
--------

- **Forgetting to update both the plain and the regrouped header.**
  Z4c has two derived headers; the source notebook for the
  regrouped variant carries the parity-aware decomposition.  A
  physics change must be applied in *both* notebooks (or in a
  shared upstream cell pair).
- **CSE-induced diff noise.**  A tiny physics edit can rewrite
  every CSE temporary in the file because temporaries are
  numbered by appearance in the canonical traversal.  Diffs that
  look enormous are usually meaningless — read the diff with
  ``--word-diff`` or ``--minimal`` if it helps, but the right
  validation is *running the tests*, not reading the diff.
- **Hand-edits drift silently.**  An emergency in-place edit to
  a ``_subexpressions.hh`` will compile, will probably even
  pass tests, but the next person who regenerates the notebook
  will overwrite your fix.  Always backport to the notebook in
  the same PR.
- **Sum ordering matters for symmetry.**  If you observe
  unexpected floating-point asymmetry in a derived quantity
  after a notebook change, the first suspect is the CSE / Add
  ordering having shifted under the new expression tree.  The
  fix lives in the notebook (regroup the SymPy ``Add`` calls
  with ``evaluate=False`` to fix the order) or in ``MyPrinter``
  (override ``_print_Add`` to enforce a parity-aware order).
- **Integer-power printing is not free.**  ``x**6`` becomes
  ``(x)*(x)*(x)*(x)*(x)*(x)`` rather than ``pow(x, 6.0)``.  This
  is intentional and is faster on most backends, but if you ever
  need ``pow``-based semantics for some reason (a true
  non-integer exponent, a special-function branch), use SymPy's
  ``sp.Pow(x, sp.Rational(1, 3))`` rather than ``x ** (1/3)`` so
  the printer sees a real non-integer exponent.
- **The ``a.out`` and ``.bak`` artefacts in ``GRACEpy`` are not authoritative.**
  Only the named ``.py`` / ``.ipynb`` notebook is.  Cleaning these
  up periodically is harmless.
