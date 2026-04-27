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
