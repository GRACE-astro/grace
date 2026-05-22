/**
 * @file grace_system.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Umbrella header for the GRACE runtime system: initialization,
 *        finalization, MPI / Kokkos / p4est lifecycle, runtime singleton,
 *        and printing utilities.  Include this to bring in all
 *        system-level facilities.
 *
 * \defgroup system GRACE runtime system
 *
 * Lifecycle management of the GRACE runtime: MPI, Kokkos, and p4est
 * initialization and shutdown; the global runtime singleton holding
 * iteration / time / parameter state; logging and print utilities.
 *
 * @version 0.1
 * @date 2024-03-18
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */
#ifndef GRACE_SYSTEM_GRACE_SYSTEM_HH
#define GRACE_SYSTEM_GRACE_SYSTEM_HH

#include <grace/system/grace_initialize.hh>
#include <grace/system/grace_finalize.hh> 
#include <grace/system/grace_runtime.hh>
#include <grace/system/mpi_runtime.hh>
#include <grace/system/kokkos_runtime.hh>
#include <grace/system/p4est_runtime.hh>
#include <grace/system/print.hh>
#include <grace/errors/error.hh>
#include <grace/errors/assert.hh>
#include <grace/system/checkpoint_handler.hh>

#endif /* GRACE_SYSTEM_GRACE_SYSTEM_HH */