/**
 * @file particle_utilities.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Shared types for the GRACE particle subsystem.
 *
 * This header defines particle-related types that are usable everywhere in
 * GRACE regardless of whether the particle subsystem is built. Particle
 * storage and kernels live under include/grace/particles/ and are guarded by
 * GRACE_ENABLE_PARTICLES.
 *
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 */
#ifndef GRACE_UTILS_PARTICLE_UTILITIES_HH
#define GRACE_UTILS_PARTICLE_UTILITIES_HH

#include <grace_config.h>

#include <cstdint>

namespace grace {

enum particle_status_flag_t : uint8_t {
    PARTICLE_DEFAULT = 0,
    PARTICLE_INSIDE_BH,
    PARTICLE_OUTSIDE_DOMAIN,
    N_PARTICLE_STATUSES
};

enum particle_drift_policy_t : uint8_t {
    DRIFT_POLICY_LAZY = 0, // tracers: rely on cached owner + ghost halo
    DRIFT_POLICY_EAGER     // MC/PIC: inline slow-path search per substep
};

} // namespace grace

#endif // GRACE_UTILS_PARTICLE_UTILITIES_HH
