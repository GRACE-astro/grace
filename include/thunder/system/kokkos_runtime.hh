/**
 * @file kokkos_runtime.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef INCLUDE_THUNDER_SYSTEM_KOKKOS_RUNTIME
#define INCLUDE_THUNDER_SYSTEM_KOKKOS_RUNTIME

#include <Kokkos_Core.hpp>

#include <thunder_config.h>

#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh> 

namespace thunder {
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Small class used to ensure <code>Kokkos</code> is initialized and finalized at appropriate times.
 * \ingroup system
 */
class kokkos_runtime_impl_t 
{
 private:
    //*****************************************************************************************************
    /**
     * @brief (Never) construct a new <code>kokkos_runtime_impl_t</code> object
     */
    kokkos_runtime_impl_t(int* argc, char* argv[]) {
        Kokkos::initialize(*argc, argv) ; 
    }
    //*****************************************************************************************************
    /**
     * @brief (Never) destroy the <code>kokkos_runtime_impl_t</code> object
     * 
     */
    ~kokkos_runtime_impl_t() {
        Kokkos::finalize() ; 
    } 
    //*****************************************************************************************************
    friend class utils::singleton_holder<kokkos_runtime_impl_t,memory::default_create> ;           //!< Give access
    friend class memory::new_delete_creator<kokkos_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access
    //*****************************************************************************************************
    static constexpr size_t longevity = KOKKOS_RUNTIME ; //!< Schedule destruction 
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief Unique proxy for Kokkos runtime.
 * \ingroup system
 */
using kokkos_runtime = utils::singleton_holder<kokkos_runtime_impl_t,memory::default_create> ;
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace thunder */

#endif /* INCLUDE_THUNDER_SYSTEM_KOKKOS_RUNTIME */
