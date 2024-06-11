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
 * @brief EOS storage class.
 * \ingroup eos 
 *
 * This class holds an instance of all possible EOS types.
 * Only the object corresponding to the active EOS type (selected
 * through the parfile) will be in a meaningful state.
 */
class eos_storage_t {

 public:
    /**
     * @brief Get the hybrid piecewise polytropic EOS object.
     * 
     * This should only be called if the active EOS type is 
     * hybrid piecewise polytrope.
     * 
     * @return decltype(auto) The hybrid piecewise polytropic EOS object.
     */
    decltype(auto) GRACE_ALWAYS_INLINE 
    get_hybrid_pwpoly() {
        return _hybrid_pwpoly ; 
    }
    /**
     * @brief Get the hybrid tabulated EOS object.
     * 
     * This should only be called if the active EOS type is 
     * hybrid tabulated.
     * 
     * @return decltype(auto) The hybrid tabulated EOS object.
     */
    decltype(auto) GRACE_ALWAYS_INLINE 
    get_hybrid_tabulated() {
        ERROR("Not implemented yet.") ;  
    }
    /**
     * @brief Get the tabulated EOS object.
     * 
     * This should only be called if the active EOS type is 
     * tabulated.
     * 
     * @return decltype(auto) The tabulated EOS object.
     */
    decltype(auto) GRACE_ALWAYS_INLINE 
    get_tabulated() {
        ERROR("Not implemented yet.") ;  
    }

 private:
    //! Longevity of EOS utils.
    static constexpr size_t longevity = GRACE_EOS_STORAGE ; 
    /**
     * @brief (Never) construct a new eos_storage_t object
     * 
     */
    eos_storage_t() ; 
    /**
     * @brief (Never) destroy the eos_storage_t object
     */
    ~eos_storage_t() {}; 

    //! The hybrid EOS object.
    hybrid_eos_t<piecewise_polytropic_eos_t> _hybrid_pwpoly ; 
    //! Give access.
    friend class utils::singleton_holder<eos_storage_t, memory::default_create>  ;
    //! Give access.
    friend class memory::new_delete_creator<eos_storage_t, memory::new_delete_allocator> ; 
} ; 
/**
 * @brief 
 * \ingroup eos
 * Singleton wrapping EOS storage in GRACE. To obtain 
 * a specific EOS object the API is:
 * \code 
 * auto const tabulated_eos = grace::eos::get().get_tabulated();
 * auto const hybrid_pwpoly_eos = grace::eos::get().get_hybrid_pwpoly();
 * auto const hybrid_tabulated_eos = grace::eos::get().get_hybrid_tabulated();
 * \endcode 
 * Note that all the EOS types are trivially copy-able and constructible on device.
 * This allows for seemless inter-operation with GPU code. This is achieved by having 
 * all data members of the concrete EOS types be either trivially copy-able or views 
 * referencing GPU memory. However, this means that in general EOS methods are \b not 
 * safe to call from host. If you really need to, a workaround needs to be found (e.g. 
 * initializing a temporary host-eos object whose views are on host).
 */
using eos = utils::singleton_holder< eos_storage_t > ; 

}

#endif 