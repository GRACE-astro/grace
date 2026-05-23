#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/utils/grace_utils.hh>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

// Exhaustive reflection-BC unit test on the octant [xmin,xmax]^3 with
// reflection symmetries on all three axes (xmin = ymin = zmin = the
// reflection planes).  Coverage:
//
//   1. Centered scalars   (e.g. DENS, ALP, CHI, ...)       parity all +1
//   2. Centered vectors   (e.g. SX/SY/SZ, BETAX/Y/Z)       parity -1 on own axis
//   3. Centered symmetric tensors   (GAMMA_TILDE_*, A_TILDE_*)
//                                                          per-component table
//   4. Face-staggered B   (BSX in face-x view, BSY in face-y view, BSZ in
//                          face-z view)                     pseudovector
//
// Strategy (same as test_bc_unigrid.cpp):
//   1. Fill every cell (interior + ghost) of every view with a linear
//      polynomial of its physical coordinates.  Degree-1 is preserved
//      bit-exactly by every AMR prolongation operator GRACE supports,
//      so the same test passes on unigrid AND on FMR.
//   2. NaN-poison ghost slots so any cell the BC kernel fails to write
//      shows up as a bit-exact mismatch.
//   3. amr::apply_boundary_conditions()
//   4. Walk every cell on the host.  For any cell at p[axis] < domain_min
//      (a reflection ghost), compute
//          expected = parity_product * h_func(reflected_coords)
//      and bit-exact-assert against the stored value.
//
// Corner-of-BCs caveat:
//   A reflection ghost at, say, x<0 AND y>ymax has its reflection source
//   at (|x|, y>ymax) — which is itself a HIGH-y ghost, filled NOT by the
//   polynomial but by whichever high-side BC ran first (outflow_0,
//   extrap_3, sommerfeld).  Bit-exact comparison against
//   parity * h_func(...) is invalid for those cells.  We SKIP cells whose
//   non-reflected axes have p > domain_max.  This is conservative — a
//   finer test could clamp the source to the last interior cell value
//   for outflow_0 vars, or use h_func directly for extrap_3 vars (which
//   preserves linear polynomials bit-exactly).
//
// Polynomial choice: linear (no cross terms).  This is the highest degree
// preserved bit-exactly by ALL of GRACE's AMR operators, so the same test
// passes on FMR with tol=0.

namespace {

// Linear polynomial of physical coords whose coefficients depend on the
// variable's unique id.  Linear is preserved bit-exactly by every AMR
// prolongation/restriction operator (Lagrange-4, second-order, etc.) — they
// collapse to identity on degree-1 inputs.  Making the coefficients vary per
// variable means any cross-variable contamination (e.g. reflection reads
// from the wrong View, wrong slot of the same View, wrong staggering)
// surfaces as a numerical mismatch instead of accidentally matching the
// other variable's polynomial.
inline double h_func_for_var(int var_id, double x, double y, double z)
{
    double const a = 8.5  + 0.13 * var_id;
    double const b = -5.1 - 0.07 * var_id;
    double const c = 2.0  + 0.19 * var_id;
    double const d = -3.14 + 0.31 * var_id;
    return a * x + b * y + c * z + d;
}

// Per-axis reflection parity for a single variable component.  parity[k]
// is the sign change when reflecting along axis k.
struct axis_parity_t { double p[3]; };

inline axis_parity_t scalar_parity()  { return {{+1.0, +1.0, +1.0}}; }

// Centered vector V^k: parity = -1 along axis k, +1 along the others.
inline axis_parity_t vector_parity(int comp)
{
    axis_parity_t out = {{+1.0, +1.0, +1.0}};
    out.p[comp] = -1.0;
    return out;
}

// Centered symmetric tensor T^{ij}.  Under reflection along axis a:
// parity = (-1)^(#axis-matches), so off-diagonals (xy, xz, yz) get -1
// along both of their indices, diagonals (xx, yy, zz) get all +1.
// comp_num convention (matches amr_ghosts.cpp:114-132):
//   0 = xx, 1 = xy, 2 = xz, 3 = yy, 4 = yz, 5 = zz
inline axis_parity_t sym_tensor_parity(int comp_num)
{
    axis_parity_t out = {{+1.0, +1.0, +1.0}};
    if (comp_num == 1) { out.p[0] = -1.0; out.p[1] = -1.0; }   // xy
    if (comp_num == 2) { out.p[0] = -1.0; out.p[2] = -1.0; }   // xz
    if (comp_num == 4) { out.p[1] = -1.0; out.p[2] = -1.0; }   // yz
    // 0 (xx), 3 (yy), 5 (zz) stay all +1
    return out;
}

// Face-staggered B (a pseudovector).  B^d is even under d-reflection,
// odd under the other two.  Matches phys_bc_kernels.hh:382-393.
inline axis_parity_t face_staggered_B_parity(grace::var_staggering_t stag)
{
    switch (stag) {
        case grace::STAG_FACEX: return {{+1.0, -1.0, -1.0}};
        case grace::STAG_FACEY: return {{-1.0, +1.0, -1.0}};
        case grace::STAG_FACEZ: return {{-1.0, -1.0, +1.0}};
        default: return {{+1.0, +1.0, +1.0}};
    }
}

// Look up a centered evolved variable's parity from its registered props.
inline axis_parity_t parity_for_centered_var(int iv)
{
    auto const& name  = grace::variables::detail::_varnames[iv];
    auto const& props = grace::variables::detail::_varprops[name];
    if (props.is_vector) return vector_parity(props.comp_num);
    if (props.is_tensor) return sym_tensor_parity(props.comp_num);
    return scalar_parity();
}

// Reduce the per-axis parity table to a single scalar sign for a cell that
// is reflected along the indicated subset of axes.
inline double parity_product(axis_parity_t const& p,
                             bool rx, bool ry, bool rz)
{
    double prod = 1.0;
    if (rx) prod *= p.p[0];
    if (ry) prod *= p.p[1];
    if (rz) prod *= p.p[2];
    return prod;
}

// Init all (i,j,k,q) entries of `view` for variable index `var_idx` with
// h_func evaluated at the cell's physical coordinates.  `cc` is the
// within-cell logical position appropriate for the staggering:
//   {0.5,0.5,0.5} → cell centre (centered variables)
//   {0.0,0.5,0.5} → face-x staggered
//   {0.5,0.0,0.5} → face-y staggered
//   {0.5,0.5,0.0} → face-z staggered
template <typename ViewT, typename CoordSysT>
void init_view_polynomial(
    ViewT&                                                  view,
    std::array<double, GRACE_NSPACEDIM> const&              cc,
    int                                                      var_idx,
    int                                                      var_id,
    long Nx, long Ny, long Nz, long nq,
    CoordSysT&                                              cs)
{
    auto h = Kokkos::create_mirror_view(view);
    for (long q = 0; q < nq; ++q)
    for (long k = 0; k < Nz; ++k)
    for (long j = 0; j < Ny; ++j)
    for (long i = 0; i < Nx; ++i) {
        auto p = cs.get_physical_coordinates(
            {VEC(static_cast<size_t>(i),
                 static_cast<size_t>(j),
                 static_cast<size_t>(k))},
            static_cast<size_t>(q), cc, /*include_gzs*/ true);
        h(VEC(i, j, k), var_idx, q) = h_func_for_var(var_id, p[0], p[1], p[2]);
    }
    Kokkos::deep_copy(view, h);
}

// NaN-poison ghost slots so any unfilled ghost surfaces as a bit-exact
// mismatch in the check (NaN == anything is false).
template <typename ViewT>
void nan_poison_ghosts(
    ViewT&                                                  view,
    int                                                      var_idx,
    long Nx, long Ny, long Nz, long nq,
    int ngz)
{
    auto h = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h, view);
    for (long q = 0; q < nq; ++q)
    for (long k = 0; k < Nz; ++k)
    for (long j = 0; j < Ny; ++j)
    for (long i = 0; i < Nx; ++i) {
        bool const ghost = (i < ngz) || (i >= Nx - ngz)
                        || (j < ngz) || (j >= Ny - ngz)
#ifdef GRACE_3D
                        || (k < ngz) || (k >= Nz - ngz)
#endif
                        ;
        if (ghost) {
            h(VEC(i, j, k), var_idx, q) =
                std::numeric_limits<double>::quiet_NaN();
        }
    }
    Kokkos::deep_copy(view, h);
}

// Check that every reflection ghost cell (p[axis] < domain_min on at
// least one axis, and no axis has p > domain_max) holds the expected
// value parity_product * h_func(reflected_coords).
//
// Returns the number of cells checked, and writes a one-line summary to
// stdout.  Per-cell mismatches are reported via Catch2 INFO + CHECK.
template <typename ViewT, typename CoordSysT>
size_t check_reflection_ghosts(
    ViewT const&                                            view,
    char const*                                              tag,
    std::array<double, GRACE_NSPACEDIM> const&              cc,
    int                                                      var_idx,
    int                                                      var_id,
    axis_parity_t const&                                    parity,
    double                                                   tol,
    CoordSysT&                                              cs,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    long Nx, long Ny, long Nz, long nq)
{
    auto h = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h, view);

    size_t n_checked = 0, n_skipped_corner = 0, n_failed = 0;
    double max_err = 0.0;

    for (long q = 0; q < nq; ++q)
    for (long k = 0; k < Nz; ++k)
    for (long j = 0; j < Ny; ++j)
    for (long i = 0; i < Nx; ++i) {
        auto p = cs.get_physical_coordinates(
            {VEC(static_cast<size_t>(i),
                 static_cast<size_t>(j),
                 static_cast<size_t>(k))},
            static_cast<size_t>(q), cc, /*include_gzs*/ true);

        bool const refl_x = (p[0] < xmin);
        bool const refl_y = (p[1] < ymin);
        bool const refl_z = (p[2] < zmin);
        if (!(refl_x || refl_y || refl_z)) continue;

        // SKIP cells whose non-reflected axes land in the HIGH ghost
        // band (p > domain_max).  Their reflection source is itself a
        // high-side ghost — value is set by outflow/extrap/sommerfeld,
        // not by h_func, so bit-exact compare against parity*h_func
        // would be invalid.  See file header for the clamp-extension.
        bool const high_x = (p[0] > xmax);
        bool const high_y = (p[1] > ymax);
        bool const high_z = (p[2] > zmax);
        if (high_x || high_y || high_z) {
            ++n_skipped_corner;
            continue;
        }

        // Mirror the reflected axes about their respective domain edges.
        double const xr = refl_x ? (2.0 * xmin - p[0]) : p[0];
        double const yr = refl_y ? (2.0 * ymin - p[1]) : p[1];
        double const zr = refl_z ? (2.0 * zmin - p[2]) : p[2];

        double const sign     = parity_product(parity, refl_x, refl_y, refl_z);
        double const expected = sign * h_func_for_var(var_id, xr, yr, zr);
        double const got      = h(VEC(i, j, k), var_idx, q);

        ++n_checked;
        double const err = std::abs(got - expected);
        max_err = std::max(max_err, err);

        INFO("var=" << tag
             << " q=" << q
             << " ijk=(" << i << "," << j << "," << k << ")"
             << " phys=(" << p[0] << "," << p[1] << "," << p[2] << ")"
             << " refl=(" << refl_x << "," << refl_y << "," << refl_z << ")"
             << " mirror=(" << xr << "," << yr << "," << zr << ")"
             << " parity_x=" << parity.p[0]
             << " parity_y=" << parity.p[1]
             << " parity_z=" << parity.p[2]
             << " sign=" << sign
             << " expected=" << expected
             << " got=" << got
             << " err=" << err);

        bool const ok = (tol == 0.0) ? (got == expected) : (err <= tol);
        if (!ok) ++n_failed;
        CHECK(ok);
    }

    std::cout << "  " << tag
              << ": checked=" << n_checked
              << " skipped_corners=" << n_skipped_corner
              << " failed=" << n_failed
              << " max_err=" << max_err << std::endl;
    return n_checked;
}

}  // namespace

// =============================================================================
// Main test case: octant unigrid OR octant FMR.  The same TEST_CASE works for
// both because reflection is a bit-exact index-mirror copy regardless of
// AMR level (the FP-arithmetic operators in the BC pipeline are outflow /
// extrap / sommerfeld on HIGH faces, which the check function skips).
//
// Behaviour controlled by the yaml: the test reads xmin/xmax/... and the
// reflection_symmetries flags, refuses to run if reflection isn't on, and
// uses tol=0 (bit-exact) in every variant.
// =============================================================================
TEST_CASE("Reflection BC: octant ghost-zone fill (bit-exact across all "
          "variable types)", "[boundaries][reflection]")
{
    using namespace grace;
    using namespace grace::variables;

    auto& state = variable_list::get().getstate();
    auto& stag  = variable_list::get().getstaggeredstate();
    auto& cs    = coordinate_system::get();

    long nx, ny, nz;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents();
    long const nq  = static_cast<long>(amr::get_local_num_quadrants());
    int  const ngz = amr::get_n_ghosts();

    auto params = config_parser::get()["amr"];
    double const xmin = params["xmin"].as<double>();
    double const ymin = params["ymin"].as<double>();
    double const zmin = params["zmin"].as<double>();
    double const xmax = params["xmax"].as<double>();
    double const ymax = params["ymax"].as<double>();
    double const zmax = params["zmax"].as<double>();

    REQUIRE(params["reflection_symmetries"]["x"].as<bool>());
    REQUIRE(params["reflection_symmetries"]["y"].as<bool>());
    REQUIRE(params["reflection_symmetries"]["z"].as<bool>());

    long const Nx        = nx + 2 * ngz;
    long const Ny        = ny + 2 * ngz;
    long const Nz        = nz + 2 * ngz;
    long const Nx_facex  = Nx + 1, Ny_facex = Ny, Nz_facex = Nz;
    long const Nx_facey  = Nx,     Ny_facey = Ny + 1, Nz_facey = Nz;
    long const Nx_facez  = Nx,     Ny_facez = Ny,     Nz_facez = Nz + 1;

    int const nv_centered = get_n_evolved();

    std::cout << "Reflection BC test: nx,ny,nz,ngz = "
              << nx << "," << ny << "," << nz << "," << ngz
              << "  nq=" << nq
              << "  nv_centered=" << nv_centered << std::endl;

    // Unique per-variable polynomial id.  Centered vars use their evolved
    // index; face-staggered B uses an offset past nv_centered so each
    // (View, var_idx) pair gets a distinct polynomial.  Any cross-variable
    // contamination (reflection reads from the wrong view or the wrong slot
    // of the right view) now surfaces as a numerical mismatch.
    int const id_BSX = nv_centered + 0;
    int const id_BSY = nv_centered + 1;
    int const id_BSZ = nv_centered + 2;

    // ---- INITIALISE ALL VIEWS WITH h_func AT EACH CELL'S PHYSICAL COORD ----
    for (int iv = 0; iv < nv_centered; ++iv) {
        init_view_polynomial(state, {VEC(0.5, 0.5, 0.5)}, iv, /*var_id*/ iv,
                             Nx, Ny, Nz, nq, cs);
    }
    init_view_polynomial(stag.face_staggered_fields_x, {VEC(0.0, 0.5, 0.5)},
                         BSX_, id_BSX, Nx_facex, Ny_facex, Nz_facex, nq, cs);
    init_view_polynomial(stag.face_staggered_fields_y, {VEC(0.5, 0.0, 0.5)},
                         BSY_, id_BSY, Nx_facey, Ny_facey, Nz_facey, nq, cs);
#ifdef GRACE_3D
    init_view_polynomial(stag.face_staggered_fields_z, {VEC(0.5, 0.5, 0.0)},
                         BSZ_, id_BSZ, Nx_facez, Ny_facez, Nz_facez, nq, cs);
#endif

    // ---- NaN-POISON GHOSTS ----
    for (int iv = 0; iv < nv_centered; ++iv) {
        nan_poison_ghosts(state, iv, Nx, Ny, Nz, nq, ngz);
    }
    nan_poison_ghosts(stag.face_staggered_fields_x, BSX_,
                      Nx_facex, Ny_facex, Nz_facex, nq, ngz);
    nan_poison_ghosts(stag.face_staggered_fields_y, BSY_,
                      Nx_facey, Ny_facey, Nz_facey, nq, ngz);
#ifdef GRACE_3D
    nan_poison_ghosts(stag.face_staggered_fields_z, BSZ_,
                      Nx_facez, Ny_facez, Nz_facez, nq, ngz);
#endif

    // ---- APPLY BCs ----
    amr::apply_boundary_conditions();

    // ---- CHECK ----
    // Refraction reflection is bit-exact at the ghost cells, even with FMR,
    // because the kernel is an index-mirror copy with a parity sign.  The
    // FP arithmetic in the BC pipeline only happens at HIGH-side ghosts
    // (outflow/extrap/sommerfeld), which the check function skips.
    double const tol = 0.0;

    // Single loop over every centered evolved variable.  parity_for_centered_var
    // looks up the var's registered props (is_vector / is_tensor / comp_num)
    // and returns the correct parity table for scalar / vector / symmetric-
    // tensor components.  This avoids the trap that
    // get_vector_state_variables_indices() / get_tensor_state_variables_indices()
    // each return ONLY the first index of every vector / tensor block.
    //
    // SKIP gamma_tilde and A_tilde under Z4 metric evolution: the BC kernel
    // correctly writes their reflection ghosts, but the algebraic-constraint
    // follow-up pass (impose_algebraic_constraintz_z4c, phys_bc_kernels.hh:299)
    // runs immediately after on every centered ghost cell.  It divides
    // gamma_tilde by cbrt(det γ̃) and adjusts A_tilde to be trace-free — on
    // our polynomial init det γ̃ = 0 generically, producing inf/nan that
    // overwrite the clean reflection result.  Testing those variables requires
    // a constraint-satisfying init (γ̃ = identity + traceless A) and is a
    // separate check.
    auto const skip_for_z4c_constraint = [](std::string const& name) {
        return name.rfind("gamma_tilde", 0) == 0
            || name.rfind("A_tilde",     0) == 0;
    };

    std::cout << "[reflection BC test] All centered evolved variables"
              << std::endl;
    size_t n_total = 0, n_skipped_vars = 0;
    for (int iv = 0; iv < nv_centered; ++iv) {
        auto const& name = variables::detail::_varnames[iv];
        if (skip_for_z4c_constraint(name)) {
            std::cout << "  " << name
                      << ": SKIPPED (clobbered by Z4c algebraic-constraint "
                         "enforcement on polynomial init)" << std::endl;
            ++n_skipped_vars;
            continue;
        }
        n_total += check_reflection_ghosts(
            state, name.c_str(),
            {VEC(0.5, 0.5, 0.5)}, iv, /*var_id*/ iv,
            parity_for_centered_var(iv), tol, cs,
            xmin, ymin, zmin, xmax, ymax, zmax,
            Nx, Ny, Nz, nq);
    }

    std::cout << "[reflection BC test] Face-staggered B" << std::endl;
    n_total += check_reflection_ghosts(
        stag.face_staggered_fields_x, "BSX",
        {VEC(0.0, 0.5, 0.5)}, BSX_, id_BSX,
        face_staggered_B_parity(STAG_FACEX), tol, cs,
        xmin, ymin, zmin, xmax, ymax, zmax,
        Nx_facex, Ny_facex, Nz_facex, nq);
    n_total += check_reflection_ghosts(
        stag.face_staggered_fields_y, "BSY",
        {VEC(0.5, 0.0, 0.5)}, BSY_, id_BSY,
        face_staggered_B_parity(STAG_FACEY), tol, cs,
        xmin, ymin, zmin, xmax, ymax, zmax,
        Nx_facey, Ny_facey, Nz_facey, nq);
#ifdef GRACE_3D
    n_total += check_reflection_ghosts(
        stag.face_staggered_fields_z, "BSZ",
        {VEC(0.5, 0.5, 0.0)}, BSZ_, id_BSZ,
        face_staggered_B_parity(STAG_FACEZ), tol, cs,
        xmin, ymin, zmin, xmax, ymax, zmax,
        Nx_facez, Ny_facez, Nz_facez, nq);
#endif

    std::cout << "[reflection BC test] TOTAL: " << n_total
              << " reflection-ghost cells verified" << std::endl;
    REQUIRE(n_total > 0);
}
