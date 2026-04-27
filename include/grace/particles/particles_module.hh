/**
 * @file particles_module.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Top-level entry point for the GRACE particle subsystem.
 *
 * The whole module compiles out when GRACE_ENABLE_PARTICLES is undefined.
 * When enabled, callers should still respect the runtime config flag
 * particles.enabled before invoking module APIs that mutate state.
 *
 * See doc/design/particles.md for the full architecture.
 */
#ifndef GRACE_PARTICLES_PARTICLES_MODULE_HH
#define GRACE_PARTICLES_PARTICLES_MODULE_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_storage.hh>

#include <cstddef>

namespace grace {
namespace particles {

class particles_module_impl_t;

class particles_module_t {
  public:
    static particles_module_t& get();

    void initialize();
    void finalize();

    bool        enabled() const noexcept;
    std::size_t local_count() const noexcept;

    /// One advance per full RK step. Sample fluid SRC state at current
    /// tracer positions and push by dt (Forward Euler at t^n).
    /// No-op when the module is disabled or has zero tracers.
    ///
    /// Sub-CFL safety: the fluid CFL bounds the signal speed, and
    /// |v_fluid| <= |v_signal|, so |v|*dt < dx. Tracers cannot cross
    /// more than one cell per call, so a single push per step is safe
    /// and migration in Phase 2a only needs to handle one quad-jump
    /// per particle per step.
    ///
    /// PIC extensibility: PIC species back-react on EM fields and need
    /// proper per-substage RK with field deposition between substages.
    /// They will get a separate advance_substep(dt, dtfact, ...) family
    /// dispatched per-species; this entry point stays tracer-shaped.
    void advance_step(double dt);

    /// Direct access to the tracer container. Mostly for tests and the
    /// (forthcoming) IO layer; the RK loop itself goes through
    /// advance_step().
    tracer_container_t<>&       tracers() noexcept;
    const tracer_container_t<>& tracers() const noexcept;

  private:
    particles_module_t();
    ~particles_module_t();
    particles_module_t(const particles_module_t&) = delete;
    particles_module_t& operator=(const particles_module_t&) = delete;

    particles_module_impl_t* _impl;
};

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLES_MODULE_HH
