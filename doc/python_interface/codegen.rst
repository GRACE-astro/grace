.. _gracepy-codegen:

Code Generation
================

The ``codegen.codegen_utils`` module provides SymPy-based utilities for generating optimized C99 code from symbolic expressions. This is used to produce the ``*_subexpressions.hh`` C++ headers that GRACE compiles into GPU kernels.

The typical workflow is:

1. Define symbolic variables and derive expressions in a Jupyter notebook using SymPy.
2. Use common subexpression elimination (CSE) to reduce redundant computation.
3. Emit C99 code with a custom printer optimized for Kokkos GPU kernels.


Custom C99 Printer
********************

The ``MyPrinter`` class extends SymPy's ``C99CodePrinter`` with power-expansion optimizations that avoid expensive ``pow()`` calls on GPUs:

- Integer powers are expanded as repeated multiplication: :math:`x^3 \to x \cdot x \cdot x`
- Negative integer powers use reciprocals: :math:`x^{-2} \to 1/(x \cdot x)`
- :math:`x^{-1/2}` and :math:`x^{-3/2}` use explicit ``sqrt`` calls
- All other powers fall through to the standard ``pow()``

.. code-block:: python

    from codegen.codegen_utils import MyPrinter

    printer = MyPrinter()
    printer.doprint(x**3)     # '((x)*(x)*(x))'
    printer.doprint(x**(-2))  # '1/((x)*(x))'


Symbolic Derivatives
**********************

Helper functions create symbolic derivative placeholders that match the array layout used in GRACE's finite-difference stencils:

.. code-block:: python

    from codegen.codegen_utils import (
        derivative_matrix, der_symm_tens,
        derivative_vector, der_vec,
    )

    # Derivative of a symmetric 3x3 tensor (returns list of 3 matrices, one per direction)
    dg = der_symm_tens(gamma, "g")    # dg[0] = d_x gamma, dg[1] = d_y gamma, ...

    # Derivative of a 3-vector (returns list of 3 vectors)
    dbeta = der_vec(beta, "beta")     # dbeta[0] = d_x beta, ...

    # Single named derivative
    dgamma_dx = derivative_matrix(gamma, "gamma", "dx")
    dbeta_dy  = derivative_vector(beta, "beta", "dy")

The ``der_symm_tens`` function exploits symmetry: for a 3x3 symmetric tensor it only creates symbols for the 6 independent components per direction, stored in Voigt order.


Emitting Assignments
**********************

Once symbolic expressions have been derived, they need to be emitted as C assignment statements. The ``emit_matrix_assignments`` function handles scalars, vectors, and matrices (including symmetric ones):

.. code-block:: python

    from codegen.codegen_utils import emit_matrix_assignments

    lines = emit_matrix_assignments(
        expr,                  # SymPy Matrix expression
        printer,               # MyPrinter instance
        "Ricci",               # output variable name
        layout="flat",         # "flat" (1D array) or 2D indexing
        enforce_symmetry=True, # use Voigt storage for symmetric matrices
        addto=False,           # False: '=', True: '+='
    )


Generating Complete Functions
*******************************

The ``make_function`` utility combines CSE, code printing, and signature generation into a single call that produces a complete C++ function:

.. code-block:: python

    from codegen.codegen_utils import make_function, MyPrinter

    code = make_function(
        exprs,            # list of SymPy expressions (one per output)
        MyPrinter(),      # code printer
        "compute_Ricci",  # function name
        ABI,              # OrderedDict: input_name -> (ctype, shape)
        outputs,          # list of output variable names
        outputs_ABI,      # OrderedDict: output_name -> (ctype, shape)
    )

The generated function:

- Is decorated with ``KOKKOS_INLINE_FUNCTION`` for GPU compatibility
- Has a deterministic argument order derived from the ABI dictionaries
- Contains CSE temporaries (``double x0 = ...;``) followed by output assignments
- Optionally supports C++ template parameters via the ``template_args`` argument

ABI dictionaries
~~~~~~~~~~~~~~~~~~

The ABI (Application Binary Interface) dictionaries define the C types and shapes of each argument:

.. code-block:: python

    ABI = OrderedDict([
        ("g",      ("const double", [3, 3])),  # const double g[3][3]
        ("beta",   ("const double", [3])),      # const double beta[3]
        ("alpha",  ("const double", None)),      # const double alpha  (scalar)
    ])

    outputs_ABI = OrderedDict([
        ("Ricci",  ("double", [6])),            # double (*Ricci)[6]
    ])

Optional keyword arguments to ``make_function``:

- ``layout`` — ``"flat"`` (default) for 1D array indexing, or 2D ``[i][j]``
- ``additional_inputs`` — extra symbols not automatically detected from expression free symbols
- ``cse_order``, ``cse_optims`` — control CSE behavior
- ``template_args`` — list of ``(type, name)`` pairs for C++ template parameters
- ``global_constants`` — symbol names that should not appear in the argument list
- ``add_to_output`` — use ``+=`` instead of ``=`` for output assignments
