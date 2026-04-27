/**
 * @file particle_advance.cpp
 * @brief Implementation of the per-substep particle advance.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_advance.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {
namespace particles {

std::array<fetch_field_spec_t, 17> tracer_v1_field_specs_resolved()
{
    using namespace grace::variables;
    return {{
        {field_source::STATE,       ALP_},
        {field_source::STATE,       BETAX_},
        {field_source::STATE,       BETAY_},
        {field_source::STATE,       BETAZ_},
        {field_source::DERIVED_V_X, 0},
        {field_source::DERIVED_V_Y, 0},
        {field_source::DERIVED_V_Z, 0},
        {field_source::DERIVED_W,   0},
        {field_source::AUX,         RHO_},
        {field_source::AUX,         TEMP_},
        {field_source::AUX,         YE_},
        {field_source::AUX,         ENTROPY_},
        {field_source::AUX,         PRESS_},
        {field_source::AUX,         EPS_},
        {field_source::AUX,         BX_},
        {field_source::AUX,         BY_},
        {field_source::AUX,         BZ_},
    }};
}

template <class MemorySpace>
void splay_into_samples(tracer_container_t<MemorySpace>& tr,
                        Kokkos::View<double**, MemorySpace> fetched)
{
    using exec_space = typename MemorySpace::execution_space;
    const std::size_t n = tr.size();

    auto sample_alpha   = tr.sample_alpha;
    auto sample_beta    = tr.sample_beta;
    auto sample_v       = tr.sample_v;
    auto sample_W       = tr.sample_W;
    auto sample_rho     = tr.sample_rho;
    auto sample_temp    = tr.sample_temp;
    auto sample_ye      = tr.sample_ye;
    auto sample_entropy = tr.sample_entropy;
    auto sample_press   = tr.sample_press;
    auto sample_eps     = tr.sample_eps;
    auto sample_B       = tr.sample_B;

    Kokkos::parallel_for("particle_splay_samples",
        Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            sample_alpha(i)     = fetched(i, 0);
            sample_beta(i, 0)   = fetched(i, 1);
            sample_beta(i, 1)   = fetched(i, 2);
            sample_beta(i, 2)   = fetched(i, 3);
            sample_v(i, 0)      = fetched(i, 4);
            sample_v(i, 1)      = fetched(i, 5);
            sample_v(i, 2)      = fetched(i, 6);
            sample_W(i)         = fetched(i, 7);
            sample_rho(i)       = fetched(i, 8);
            sample_temp(i)      = fetched(i, 9);
            sample_ye(i)        = fetched(i, 10);
            sample_entropy(i)   = fetched(i, 11);
            sample_press(i)     = fetched(i, 12);
            sample_eps(i)       = fetched(i, 13);
            sample_B(i, 0)      = fetched(i, 14);
            sample_B(i, 1)      = fetched(i, 15);
            sample_B(i, 2)      = fetched(i, 16);
        });
    Kokkos::fence();
}

// Explicit instantiation for default_space (only one we use right now).
template void splay_into_samples<grace::default_space>(
    tracer_container_t<grace::default_space>&,
    Kokkos::View<double**, grace::default_space>);

void advance_substep(
    MPI_Comm comm,
    double dt,
    double dtfact,
    Kokkos::View<double*[3], grace::default_space> src_pos,
    Kokkos::View<double*[3], grace::default_space> dst_pos,
    tracer_container_t<grace::default_space>& tr)
{
    using exec_space = typename grace::default_space::execution_space;
    const std::size_t n = tr.size();
    // No early return on n == 0: fetch_at_positions builds two
    // distribution_plan_t (each posts MPI_Alltoall + Ialltoallv) and is
    // collective. A sibling rank with non-empty tr would deadlock if we
    // skipped it here. The kernels below all loop over [0, n) and are
    // empty-safe; the fetched view of size (0, 17) is a valid Kokkos View.

    // -----------------------------------------------------------------
    // 1. Fetch v1 field set at src_pos.
    // -----------------------------------------------------------------
    const auto specs = tracer_v1_field_specs_resolved();
    Kokkos::View<double**, grace::default_space> fetched("particle_fetched",
                                                         n, 17);
    fetch_at_positions<17>(comm, n, specs,
                           tr.owner_rank, tr.owner_local_quad,
                           src_pos, fetched);

    // -----------------------------------------------------------------
    // 2. Splay into per-field sample slices.
    // -----------------------------------------------------------------
    splay_into_samples<grace::default_space>(tr, fetched);

    // -----------------------------------------------------------------
    // 3. Position update: dst = src + dtfact * dt * (alpha * v - beta).
    //    src_pos and dst_pos are allowed to alias.
    // -----------------------------------------------------------------
    const double scale = dtfact * dt;
    auto sample_alpha = tr.sample_alpha;
    auto sample_beta  = tr.sample_beta;
    auto sample_v     = tr.sample_v;

    Kokkos::parallel_for("particle_push",
        Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double alpha = sample_alpha(i);
            const double bx    = sample_beta(i, 0);
            const double by    = sample_beta(i, 1);
            const double bz    = sample_beta(i, 2);
            const double vx    = sample_v(i, 0);
            const double vy    = sample_v(i, 1);
            const double vz    = sample_v(i, 2);
            const double sx = src_pos(i, 0);
            const double sy = src_pos(i, 1);
            const double sz = src_pos(i, 2);
            dst_pos(i, 0) = sx + scale * (alpha * vx - bx);
            dst_pos(i, 1) = sy + scale * (alpha * vy - by);
            dst_pos(i, 2) = sz + scale * (alpha * vz - bz);
        });
    Kokkos::fence();
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
