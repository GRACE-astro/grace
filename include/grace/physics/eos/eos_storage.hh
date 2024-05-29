/**
 * @file eos_storage.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#ifndef GRACE_PHYSICS_EOS_EOS_STORAGE_HH
#define GRACE_PHYSICS_EOS_EOS_STORAGE_HH

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>

#include <Kokkos_Core.hpp>

namespace grace {
//**************************************************************************************************
/**
 * \defgroup eos Equations of state.
 * \details See the [full documentation](doc/eos.md) for details.
 */
//**************************************************************************************************
/**
 * @brief EOS storage class
 * \ingroup eos 
 *
 * 
 */
class eos_storage_t {

 public:

    decltype(auto) GRACE_ALWAYS_INLINE 
    get_hybrid_pwpoly() {
        return _hybrid_pwpoly ; 
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    get_hybrid_tabulated() {
        ERROR("Not implemented yet.") ;  
    }

    decltype(auto) GRACE_ALWAYS_INLINE 
    get_tabulated() {
        ERROR("Not implemented yet.") ;  
    }
 private:

    static constexpr size_t longevity = GRACE_EOS_STORAGE ; 

    eos_storage_t() ; 

    ~eos_storage_t() {}; 

    hybrid_eos_t<piecewise_polytropic_eos_t> _hybrid_pwpoly ; 

    friend class utils::singleton_holder<eos_storage_t, memory::default_create>  ;
    friend class memory::new_delete_creator<eos_storage_t, memory::new_delete_allocator> ; 

} ; 

using eos = utils::singleton_holder< eos_storage_t > ; 

}

#endif 