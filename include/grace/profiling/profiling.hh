/**
 * @file profiling.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-11
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
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
#ifndef GRACE_PROFILING_PROFILING_HH
#define GRACE_PROFILING_PROFILING_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/profiling/gpu_profiling.hh>
#include <grace/profiling/profiling_runtime.hh>

#include <Kokkos_Core.hpp>
//*********************************************************************************************************************
//*********************************************************************************************************************
#ifdef GRACE_ENABLE_PROFILING
//*********************************************************************************************************************
#define GRACE_KOKKOS_PROFILING_PUSH_REGION(name) Kokkos::Profiling::pushRegion(name)
//*********************************************************************************************************************
#define GRACE_DEVICE_PROFILING_PUSH_REGION(name) \
do {\
grace::profiling_runtime::get().push_device_region(name); \
Kokkos::Profiling::pushRegion(name);\
} while(false)
//*********************************************************************************************************************
#define GRACE_HOST_PROFILING_PUSH_REGION(name) \
grace::profiling_runtime::get().push_host_region(name)
//*********************************************************************************************************************
#define GRACE_PROFILING_PUSH_REGION(name) \
do {\
grace::profiling_runtime::get().push_host_region(name);\
grace::profiling_runtime::get().push_device_region(name);\
Kokkos::Profiling::pushRegion(name);\
} while(false)
//*********************************************************************************************************************
//*********************************************************************************************************************
#define GRACE_KOKKOS_PROFILING_POP_REGION Kokkos::Profiling::popRegion()
//*********************************************************************************************************************
#define GRACE_DEVICE_PROFILING_POP_REGION \
do{\
grace::profiling_runtime::get().pop_device_region(name); \
Kokkos::Profiling::popRegion();\
} while(false)
//*********************************************************************************************************************
#define GRACE_HOST_PROFILING_POP_REGION \
grace::profiling_runtime::get().pop_host_region()
//*********************************************************************************************************************
#define GRACE_PROFILING_POP_REGION \
do{\
grace::profiling_runtime::get().pop_host_region();\
grace::profiling_runtime::get().pop_device_region();\
Kokkos::Profiling::popRegion();\
} while(false)
//*********************************************************************************************************************
#else
//*********************************************************************************************************************
#define GRACE_KOKKOS_PROFILING_PUSH_REGION(name) Kokkos::Profiling::pushRegion(name)
#define GRACE_DEVICE_PROFILING_PUSH_REGION(name) Kokkos::Profiling::pushRegion(name)
#define GRACE_HOST_PROFILING_PUSH_REGION(name) Kokkos::Profiling::pushRegion(name)
#define GRACE_PROFILING_PUSH_REGION(name) Kokkos::Profiling::pushRegion(name)
#define GRACE_KOKKOS_PROFILING_POP_REGION Kokkos::Profiling::popRegion()
#define GRACE_DEVICE_PROFILING_POP_REGION Kokkos::Profiling::popRegion()
#define GRACE_HOST_PROFILING_POP_REGION Kokkos::Profiling::popRegion()
#define GRACE_PROFILING_POP_REGION Kokkos::Profiling::popRegion()
//*********************************************************************************************************************
#endif
//*********************************************************************************************************************
//*********************************************************************************************************************

#endif /* GRACE_PROFILING_PROFILING_HH */