/**
 * @file forest.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
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

#ifndef THUNDER_AMR_FOREST_HH
#define THUNDER_AMR_FOREST_HH

#include <thunder_config.h> 

#include <thunder/utils/inline.h>
#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/sc_wrappers.hh>

#include <thunder/amr/p4est_headers.hh>

#include <thunder/amr/tree.hh>
#include <thunder/amr/quadrant.hh>

#include <thunder/parallel/mpi_wrappers.hh> 

namespace thunder { namespace amr {

/**
 * @brief Wrapper around the p4est structure.
 * 
 * @tparam THUNDER_NSPACEDIM Number of space dimensions.
 * 
 */
class forest_impl_t 
{
 private:
    p4est_t * _p4est ; //!< Pointer to the p4est object 

 public: 
    /**
     * @brief Get the trees object
     * 
     * @return sc_array_view_t<p4est_tree_t> Array view containing the local trees on this rank.
     * NB: the entries in this array are only valid starting at <code>first_local_tree</code> 
     *     and up to <code>last_local_tree</code> 
     */
    sc_array_view_t<p4est_tree_t> THUNDER_ALWAYS_INLINE
    trees() const { return sc_array_view_t<p4est_tree_t>(_p4est->trees) ; } ; 
    /**
     * @brief Get first valid index of tree array on this rank.
     */
    size_t THUNDER_ALWAYS_INLINE 
    first_local_tree() const { return static_cast<size_t>( _p4est->first_local_tree) ; }
    /**
     * @brief Get last valid index of tree array on this rank.
     */
    size_t THUNDER_ALWAYS_INLINE
    last_local_tree() const { return static_cast<size_t>( _p4est->last_local_tree) ; }
    /**
     * @brief Get number of local quadrants on this rank.
     */
    size_t THUNDER_ALWAYS_INLINE
    local_num_quadrants() const { return static_cast<size_t>( _p4est->local_num_quadrants) ; }
    /**
     * @brief Get pointer to underlying p4est object. 
     */
    THUNDER_ALWAYS_INLINE p4est_t* 
    get() const { return _p4est ; }

 private:

    static constexpr unsigned int longevity = AMR_FOREST ; //!< Longevity of p4est object. 

    /**
     * @brief Never construct a new forest_impl_t object
     */
    forest_impl_t() ; 
    /**
     * @brief Never destroy the forest_impl_t object
     * 
     */
    ~forest_impl_t() ; 
    
    friend class utils::singleton_holder<forest_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<forest_impl_t, memory::new_delete_allocator> ; //!< Give access

} ; 

using forest = utils::singleton_holder<forest_impl_t > ; 

}} /* thunder::amr */
#endif /* THUNDER_AMR_FOREST_HH */
