#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>

#include <array>
#include <iostream>
#include <limits>

namespace {

// True iff the physical coordinate `p` lies outside the active simulation
// domain. Used to skip cells that get filled by a phys-BC kernel — those
// involve floating-point arithmetic (extrap_3, sommerfeld) so bit-exactness
// is not guaranteed even for a linear test function.
inline bool is_phys_boundary(std::array<double, GRACE_NSPACEDIM> const& p)
{
    auto params = grace::config_parser::get()["amr"];
#ifdef GRACE_CARTESIAN_COORDINATES
    double const xmin = params["xmin"].as<double>();
    double const ymin = params["ymin"].as<double>();
    double const xmax = params["xmax"].as<double>();
    double const ymax = params["ymax"].as<double>();
    bool out = (p[0] < xmin) || (p[0] > xmax)
            || (p[1] < ymin) || (p[1] > ymax);
#ifdef GRACE_3D
    double const zmin = params["zmin"].as<double>();
    double const zmax = params["zmax"].as<double>();
    out = out || (p[2] < zmin) || (p[2] > zmax);
#endif
    return out;
#else
    auto const Ro = params["outer_region_radius"].as<double>();
    double r2 = math::int_pow<2>(p[0])
              + math::int_pow<2>(p[1])
#ifdef GRACE_3D
              + math::int_pow<2>(p[2])
#endif
              ;
    return r2 > Ro * Ro;
#endif
}

}  // namespace

// Bit-exact ghost-zone fill on a uniformly refined grid.
//
// Strategy: initialise interior cells with a closed-form linear function of the
// physical coordinates, NaN out the ghost zones, run apply_boundary_conditions(),
// then check every ghost cell whose physical position lies INSIDE the domain
// (i.e. cells filled by inter-quadrant copy or by MPI pack/unpack — both pure
// `=` assignment with no FP arithmetic) is bit-exactly equal to the function
// evaluated at the same physical position.
//
// Cells whose physical position lies outside the domain (i.e. filled by a
// phys-BC kernel that does FP arithmetic) are skipped — they would not satisfy
// bit-exactness in general.
//
// Coverage: centered scalar (DENS) plus all three face staggerings
// (FACEX/FACEY/FACEZ), including the corner ghost zones.
TEST_CASE("BC bit-exact ghost-zone fill (unigrid)", "[boundaries]")
{
    using namespace grace;
    using namespace grace::variables;

#if defined(GRACE_ENABLE_BURGERS) || defined(GRACE_ENABLE_SCALAR_ADV)
    int const DENS = U;
#else
    int const DENS = DENS_;
#endif

    auto& state = variable_list::get().getstate();
    auto& stag  = variable_list::get().getstaggeredstate();
    auto& coord_system = coordinate_system::get();

    long nx, ny, nz;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents();
    long const nq  = static_cast<long>(amr::get_local_num_quadrants());
    int  const ngz = amr::get_n_ghosts();

    std::cout << "BC bit-exact unigrid test: nx,ny,nz,ngz = "
              << nx << "," << ny << "," << nz << "," << ngz
              << " nq=" << nq << std::endl;

    auto const h_func = [&](VEC(double x, double y, double z)) {
        return EXPR(8.5 * x, -5.1 * y, +2.0 * z) - 3.14;
    };

    // Physical coordinates of a point inside the cell at index (i,j,k) of
    // local quadrant q. `cc` selects the within-cell logical position:
    //   {0.5,0.5,0.5} → cell centre        (centred fields)
    //   {0.0,0.5,0.5} → low-x face centre  (face-x staggering)
    //   {0.5,0.0,0.5} → low-y face centre  (face-y staggering)
    //   {0.5,0.5,0.0} → low-z face centre  (face-z staggering)
    auto phys = [&](VEC(long i, long j, long k), long q,
                    std::array<double, GRACE_NSPACEDIM> const& cc) {
        return coord_system.get_physical_coordinates(
            {VEC(static_cast<size_t>(i),
                 static_cast<size_t>(j),
                 static_cast<size_t>(k))},
            static_cast<size_t>(q), cc, /*include_gzs*/ true);
    };

    // Initialise an arbitrary 4-D view (i,j,k,var,q) with `h_func` at every
    // interior point and NaN at every ghost point. Shape (Nx,Ny,Nz) is the
    // full extent including ghosts; the interior runs i ∈ [ngz, Nx-ngz),
    // i.e. Nx - 2*ngz interior cells along x.
    auto init_view = [&](auto& view,
                         std::array<double, GRACE_NSPACEDIM> const& cc,
                         VEC(long Nx, long Ny, long Nz),
                         int var_idx) {
        auto h = Kokkos::create_mirror_view(view);
        for (long q = 0; q < nq; ++q) {
        for (long k = 0; k < Nz; ++k) {
        for (long j = 0; j < Ny; ++j) {
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
            } else {
                auto p = phys(VEC(i, j, k), q, cc);
                h(VEC(i, j, k), var_idx, q) =
                    h_func(VEC(p[0], p[1], p[2]));
            }
        }}}}
        Kokkos::deep_copy(view, h);
    };

    // Bit-exact check: every cell whose physical position is inside the
    // domain must equal h_func evaluated at that position. NaN cells fail
    // this check (NaN == anything is false), surfacing any ghost zone that
    // BC failed to fill.
    auto check_view = [&](auto& view,
                          char const* name,
                          std::array<double, GRACE_NSPACEDIM> const& cc,
                          VEC(long Nx, long Ny, long Nz),
                          int var_idx) {
        auto h = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(h, view);
        size_t n_checked = 0, n_failed = 0;
        for (long q = 0; q < nq; ++q) {
        for (long k = 0; k < Nz; ++k) {
        for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            auto p = phys(VEC(i, j, k), q, cc);
            if (is_phys_boundary(p)) continue;  // FP-arith path; skip

            double const expected = h_func(VEC(p[0], p[1], p[2]));
            double const got      = h(VEC(i, j, k), var_idx, q);
            ++n_checked;
            INFO("staggering=" << name
                 << " q=" << q
                 << " ijk=(" << i << "," << j << "," << k << ")"
                 << " phys=(" << p[0] << "," << p[1]
#ifdef GRACE_3D
                 << "," << p[2]
#endif
                 << ")"
                 << " expected=" << expected
                 << " got=" << got);
            bool const ok = (got == expected);
            if (!ok) ++n_failed;
            CHECK(ok);
        }}}}
        std::cout << "  " << name
                  << ": checked " << n_checked
                  << " cells, " << n_failed << " mismatches" << std::endl;
    };

    // ===== Initialise =====
    init_view(state, {VEC(0.5, 0.5, 0.5)},
              VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz), DENS);

#if defined(GRACE_ENABLE_GRMHD)
    init_view(stag.face_staggered_fields_x, {VEC(0.0, 0.5, 0.5)},
              VEC(nx + 2*ngz + 1, ny + 2*ngz, nz + 2*ngz), BSX_);
    init_view(stag.face_staggered_fields_y, {VEC(0.5, 0.0, 0.5)},
              VEC(nx + 2*ngz, ny + 2*ngz + 1, nz + 2*ngz), BSY_);
#ifdef GRACE_3D
    init_view(stag.face_staggered_fields_z, {VEC(0.5, 0.5, 0.0)},
              VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz + 1), BSZ_);
#endif
#endif

    // ===== Apply BC =====
    amr::apply_boundary_conditions();

    // ===== Check =====
    check_view(state, "CENTER", {VEC(0.5, 0.5, 0.5)},
               VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz), DENS);

#if defined(GRACE_ENABLE_GRMHD)
    check_view(stag.face_staggered_fields_x, "FACEX", {VEC(0.0, 0.5, 0.5)},
               VEC(nx + 2*ngz + 1, ny + 2*ngz, nz + 2*ngz), BSX_);
    check_view(stag.face_staggered_fields_y, "FACEY", {VEC(0.5, 0.0, 0.5)},
               VEC(nx + 2*ngz, ny + 2*ngz + 1, nz + 2*ngz), BSY_);
#ifdef GRACE_3D
    check_view(stag.face_staggered_fields_z, "FACEZ", {VEC(0.5, 0.5, 0.0)},
               VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz + 1), BSZ_);
#endif
#endif
}
